# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import os
import sys
import random
import time
import datetime
from contextlib import suppress

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.multiprocessing as mp
from torch.profiler import profile, ProfilerActivity
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .utils.patchs import ALL_PATCHS # Keep the import as it is required for patching
from .utils.utility import get_effective_batch_size, unwrap, log_mem_info, absolute_data_dir_path
from .utils.distributed import IAccelerator
from .data.data import DATASETS, load_dataset_from_config
from .data.eval import MetricEvaluator
from .models import TOKENIZER_LOADERS, LLM_LOADERS, ADAPTATER_LOADERS, FEAT_LOADERS, EXTRACTOR_LOADERS
from .models.finetuning import apply_prompt_tuning
from .models.finetuning import BiasTunerWrapper
from .optim.optimizer import create_optimizer
from .optim.scheduler import create_scheduler
from .models.utils import freeze_whole_model, unfreeze_parameters_of_layers_types, show_trainable_params_percent
from .depalmodel import Depalm, DepalmTrainer


MAX_USE_GPU = 0.5
NO_SPLIT_MODULES = ['Block', 'OPTDecoderLayer', 'FeatExtractLayerWrapper']
NORM_LAYERS = [torch.nn.LayerNorm, LlamaRMSNorm]


def log_expriment_setting(config, accelerator):
    if accelerator.is_main_process:
        accelerator.log.info(accelerator)
        accelerator.log.info(f"Running experiment inside {os.getcwd()}")
        accelerator.log.info(f"Running command {' '.join(sys.argv)}")
        log_mem_info(accelerator)

        config_yaml = OmegaConf.to_yaml(config)
        with open('experiment_config.yaml', 'w') as cfg_file:
            cfg_file.write(config_yaml)
            accelerator.log.info(f"Config:\n{config_yaml}")
    else:
        print(accelerator)

def fix_config(config):
    # Few fix for config (added for the release version of the code)
    if hasattr(config.feat_model, 'data_path'):
        config.feat_model.data_path = absolute_data_dir_path(config.feat_model.data_path)
    if hasattr(config.dataset, 'audio_model_type') and config.dataset.audio_model_type == 'auto':
        config.dataset.audio_model_type = config.feat_model.audio_model_type
    if config.dataset.get('root', None):
        config.dataset.root = absolute_data_dir_path(config.dataset.root)
    if config.dataset.get('vqa_data_path', None):
        config.dataset.vqa_data_path = absolute_data_dir_path(config.dataset.vqa_data_path)
    if config.dataset.get('data_paths', None):
        config.dataset.data_paths = [absolute_data_dir_path(p) for p in config.dataset.data_paths]
    if config.llm.get('data_path', None):
        config.llm.data_path = absolute_data_dir_path(config.llm.data_path)
    if config.llm.get('cache_dir', None):
        config.llm.cache_dir = absolute_data_dir_path(config.llm.cache_dir)

def build_depalm(config, accelerator) -> Depalm:
    fix_config(config)

    with accelerator.main_process_first():
        tokenizer = TOKENIZER_LOADERS[config.llm.loader](config.llm)
        model_text = LLM_LOADERS[config.llm.loader](config.llm, config)
        feat_model, feat_transform = FEAT_LOADERS[config.feat_model.loader](config.feat_model, config)

    if config.feat_model.load_float16:
        feat_model = feat_model.half()

    ##### Freeze & fine-tune models #####

    freeze_whole_model(model_text)
    freeze_whole_model(feat_model)

    tune_cfg = config.finetune
    BiasTunerWrapper.wrap_linears(feat_model, tune_cfg.feat_model.bias_tuning)
    BiasTunerWrapper.wrap_linears(model_text, tune_cfg.text_model.bias_tuning)

    if tune_cfg.text_model.unfreeze_norm:
        unfreeze_parameters_of_layers_types(model_text, *NORM_LAYERS, as_float32=not config.distributed.cast_float16)
    if tune_cfg.feat_model.unfreeze_norm:
        unfreeze_parameters_of_layers_types(feat_model, *NORM_LAYERS, as_float32=not config.distributed.cast_float16)

    if tune_cfg.prompt_tuning.enable and tune_cfg.prompt_tuning.init_first: # If add prompt tuning first
        apply_prompt_tuning(tune_cfg.prompt_tuning, model_text)

    ##### Build adaptors and loaders #####

    extractor_builder = EXTRACTOR_LOADERS[config.feat_model.loader]
    feat_model._global_config = config # To let the wrapper get the config, hack to not edit too much code
    feat_model = extractor_builder(
        feat_model,
        features_batch_axis=config.feat_model.features_batch_axis,
        norm=config.feat_model.norm,
    )
    if config.adapter.name in ADAPTATER_LOADERS.get(config.llm.loader, {}):
        adapter_builder = ADAPTATER_LOADERS[config.llm.loader][config.adapter.name]
    else:
        adapter_builder = ADAPTATER_LOADERS['*'][config.adapter.name]
    model_text = adapter_builder(config, model_text, feat_model)

    depalm_model = Depalm(
        config=config,
        tokenizer=tokenizer,
        model_text=model_text,
        feat_model=feat_model,
        feat_transform=feat_transform,
        accelerator=accelerator,
    )

    ##### Load model checkpoint #####
    if config.load_model:
        config.load_model = config.load_model.replace('-', '_')
        load_path = config.load_model
        accelerator.log.info(f"Loading checkpoint from {load_path}...")
        depalm_model.load(load_path, remove_in_proj=config.finetune.remove_in_proj)
    show_trainable_params_percent(accelerator.log, depalm_model, 'Depalm model')

    ##### Wrap the model (DDP / FSDP) #####
    if config.compile:
        depalm_model.feat_model = accelerator.prepare_model(depalm_model.feat_model, compile=True)
        depalm_model.model_text = accelerator.prepare_model(depalm_model.model_text, compile=True)
    else:
        accelerator.barrier()
        print(f"{accelerator} with model {type(depalm_model)}")
        depalm_model = accelerator.prepare_model(depalm_model)

    accelerator.log.info(f'Model: \n{depalm_model}')
    return depalm_model

def build_trainer(config, depalm_model):
    accelerator = unwrap(depalm_model).accelerator
    #### Data #####
    log = accelerator.log
    log.info("Loading data...")


    accelerator.set_seed(config.seed) # Reset the seed to ensure every model with the same seed will have the same data split
    data_loaders = load_dataset_from_config(
        DATASETS[config.dataset.loader], config.dataset, feat_transform=unwrap(depalm_model).feat_transform, accelerator=accelerator,
    )
    metrics = MetricEvaluator(config.dataset.metrics, config.dataset.metric_eval_tokenize)

    ##### Optimizer & scheduler #####
    log.info("Creating trainer...")
    log.info(f"Effective batch_size of {get_effective_batch_size(accelerator, config)}")

    if config.eval or config.test or config.test_on_train:
        optimizer, scheduler = None, None
    else:
        optimizer = create_optimizer(depalm_model, config.training.optimizer)
        scheduler = create_scheduler(accelerator, optimizer, config)

    return DepalmTrainer(depalm_model, optimizer=optimizer, scheduler=scheduler, data_loaders=data_loaders, metrics=metrics)

def force_cudnn_initialization(device):
    s = 32
    torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=device), torch.zeros(s, s, s, s, device=device))

def get_profiler(config):
    if not config.profiling:
        return suppress()
    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True, record_shapes=True,
    )

def run_depalm_process(process_id, world_size, master_port) -> None:
    cfg_path = str(Path(__file__).parent.parent / "config")
    @hydra.main(version_base=None, config_path=cfg_path, config_name="run")
    def run_main(config : DictConfig):
        if config.exp_name is None:
            raise ValueError("No experiment selected (use +experiment=<experiment name>)")
        Path(config.run_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(config.run_dir)

        # Slurm-specific settings
        slurm_id = os.environ.get('SLURM_JOBID', None)
        if 'SLURM_ARRAY_JOB_ID' in os.environ and 'SLURM_ARRAY_TASK_ID' in os.environ:
            slurm_id = os.environ['SLURM_ARRAY_JOB_ID'] + '_' + os.environ['SLURM_ARRAY_TASK_ID']

        # Init environment
        if config.distributed.master_port is None:
            config.distributed.master_port = master_port
        for var_name, var_value in config.env.items():
            os.environ[var_name] = str(var_value)
        torch.backends.cuda.enable_flash_sdp(True)

        # Init multiprocessing
        accelerator = IAccelerator(process_id, world_size, config.distributed)
        accelerator.set_seed(config.seed)
        log = accelerator.log
        force_cudnn_initialization(accelerator.device)

        log.info(f"slurm_id={slurm_id}")
        log_expriment_setting(config, accelerator)

        if world_size == 0: # Small fix to run tests on CPU
            config.llm.load_float16 = False
            config.feat_model.load_float16 = False
            config.training.log_freq = 1


        ##### Model #####
        log.info("Creating model...")
        depalm_model = build_depalm(config, accelerator)


        ##### Trainer #####
        trainer = build_trainer(config, depalm_model)

        log.info("Memory used when the model is on GPUs:")
        log_mem_info(accelerator)

        test_on = config.dataset.test_on.split('_')

        with get_profiler(config) as prof:
            start_time = time.time()
            if config.test_on_train:
                log.info("Evaluation only (train split)")
                trainer.evaluate('train', save_as='test_on_train')
            elif config.test:
                log.info(f"Evaluation only (test split [={test_on}])")
                trainer.evaluate(test_on, save_as='test')
            elif config.eval:
                log.info("Evaluation only (val split)")
                trainer.evaluate('val', save_as='eval')
            else:
                log.info("Training the model")
                trainer.train(test_on=test_on)
            total_time = time.time() - start_time

        if prof is not None:
            Path('profiler').mkdir(exist_ok=True)
            prof.export_chrome_trace(f'profiler/profiler_{accelerator.local_rank}_trace.json')
            if accelerator.is_main_process:
                print(prof.key_averages().table(sort_by="cuda_time_total"))


        ##### Cleanup #####
        log.info(f"Run finished, logs can be found in {os.getcwd()}")
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        log.info('Running time {}'.format(total_time_str))

    return run_main()


@hydra.main(version_base=None, config_path="../config", config_name="run")
def run_main(config : DictConfig) -> None:
    print(f"Running on: {os.environ.get('SLURM_JOB_NODELIST', 'local')}")
    print("Running args:", sys.argv)
    print(f"Experiment: {config.exp_name}")
    print(f"Running on {torch.cuda.device_count()} GPUs")

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("NO GPU, will run on a CPU")
        from hydra.core.global_hydra import GlobalHydra
        GlobalHydra.instance().clear()
        run_depalm_process(0, world_size, None)
    else:
        if not config.distributed.enable:
            world_size = 1
        master_port = 20000 + random.randint(0, 10000) # Before seeding the RNG

        mp.spawn(
            run_depalm_process,
            args=(world_size, master_port),
            nprocs=world_size,
            join=True
        )