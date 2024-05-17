# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import time
import datetime
from dataclasses import dataclass
from typing import Dict, Union
from pathlib import Path
from itertools import chain
from PIL import Image

import torch
from tqdm import tqdm
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    FullStateDictConfig,
    StateDictType,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from .models.finetuning import apply_prompt_tuning
from .utils.metrics import MetricLogger, SmoothedValue
from .utils.utility import unwrap, GlobalState
from .data.eval import MetricEvaluator
from .models.utils import freeze_whole_model, unfreeze_parameters, find_submodule_with_name, find_parameter_with_name


UNUSED_LABEL_IDX = -100
DUMP_ANSWERS_PATH = Path('answers_dump')
SAVE_MODEL_PATH = Path('checkpoints/model')
SAVE_TRAIN_STATE_PATH = Path('checkpoints/training')
SAVE_BLACKLIST = ['model_text.lm_head.weight', 'model_text.model.decoder.embeding_layer.weight']


@dataclass
class GenerationModelOutput:
    """
        Class for the output generation of a model, with additional informations about input and tokens
    """
    input_tokens: torch.LongTensor
    output_tokens: torch.LongTensor
    generated_tokens: torch.LongTensor


class Depalm(torch.nn.Module):
    def __init__(self, config, tokenizer, model_text, feat_model, feat_transform, accelerator):
        super().__init__()

        self.config = config
        self.dtype = accelerator.dtype
        self.tokenizer = tokenizer
        self.model_text = model_text
        self.feat_model = feat_model
        self.feat_transform = feat_transform
        self.accelerator = accelerator

        self._init_finetuning()

    def _init_finetuning(self):
        tune_cfg = self.config.finetune

        if tune_cfg.prompt_tuning.enable and not tune_cfg.prompt_tuning.init_first: # If add prompt tuning after
            apply_prompt_tuning(tune_cfg.prompt_tuning, self.model_text)

        if tune_cfg.train_only_first_proj:
            freeze_whole_model(self)
            adapter_connector = find_submodule_with_name(self, '.adapter_connectors')
            print('adapter_connector', adapter_connector)
            unfreeze_parameters(adapter_connector, ['0.0.weight', '0.0.bias'])

            if tune_cfg.also_train_prompt:
                find_parameter_with_name(self, '.prompt_tokens').requires_grad = True
            if tune_cfg.also_train_end_proj:
                unfreeze_parameters(adapter_connector[-1][-1], ['bias', 'weight'])
            if tune_cfg.also_train_end_proj_bias:
                unfreeze_parameters(adapter_connector[-1][-1], ['bias'])

    def tokenize_input(self, prompts, instructions, label=None):
        """
            Returns the tokenization of the input as a dictionary. Includes the label in the text if not empty.
            The schema is:
            - input_ids: token ids of the input (prompt + label)
            - attention_mask: mask added to mask the padding tokens
            - labels: training target (if label != None), set to the input_ids value or UNUSED_LABEL_IDX for each token
        """
        assert isinstance(prompts, list) and isinstance(instructions, list) and len(prompts) == len(instructions)
        assert label is None or (isinstance(label, list) and len(prompts) == len(label))
        assert all(['{prompt}' in ins for ins in instructions])

        formated_prompts = []
        for single_prompt, prompt_template in zip(prompts, instructions):
            # Format the prompt using the instruction template
            single_prompt = prompt_template.format(prompt=single_prompt)
            # Add the special answer prompt if need to be
            if self.config.llm.special_answer_prompt is not None:
                if not single_prompt.strip(): # Otherwise, already have a prompt
                    single_prompt += self.config.llm.special_answer_prompt
            # If there is no whitespace at the end, add it
            if single_prompt and single_prompt[-1] not in [' ', '\n']:
                single_prompt += ' '
            # Add the special answer token if need to be
            if self.config.llm.special_answer_token is not None:
                single_prompt += self.config.llm.special_answer_token
            # print(f'single_prompt "{single_prompt}"')
            formated_prompts.append(single_prompt)

        if label is not None: # Return data for training
            label = [
                single_label + (self.tokenizer.eos_token if self.config.llm.add_eos else '')
                for single_label in label
            ]
            qa_concatenated = [ques + ans for ques, ans in zip(formated_prompts, label)]
            qa_tokens = self.tokenizer(qa_concatenated, padding='longest', return_tensors="pt").to(self.accelerator.device)

            qa_tokens['labels'] = qa_tokens.input_ids.masked_fill(qa_tokens.input_ids == self.tokenizer.pad_token_id, UNUSED_LABEL_IDX)
            if self.config.training.loss_label_only:
                for i_qa, ques in enumerate(formated_prompts):
                    ques_tokens = self.tokenizer(ques, padding=False, return_tensors="pt").to(self.accelerator.device)
                    ques_tokens = ques_tokens['input_ids'].reshape(-1)
                    qa_tokens['labels'][i_qa][:len(ques_tokens)] = UNUSED_LABEL_IDX # Train for answers, not questions

            return qa_tokens
        else: # Return data for testing
            for prompt in formated_prompts:
                if len(prompt) != len(formated_prompts[0]):
                    raise ValueError("Can't use prompts of different size during generation, padding leads to incorrect results")
            return self.tokenizer(formated_prompts, padding='longest', return_tensors="pt").to(self.accelerator.device)

    def forward(self,
        perceptual_data=None, prompt=None, instruction=None, label=None, generate=False, # General args
        image=None, text=None, labels=None, return_dict=None, mode=None, reduction=None, # Args for compatibility with epalm
        max_length=None, do_sample=None, num_beams=None, # Unused args for compatibility with epalm code
    ) -> Union[CausalLMOutputWithPast, GenerationModelOutput]:
        # Conditions for compatibility with epalm
        if image is None:
            is_epalm = False
            assert perceptual_data is not None and prompt is not None and instruction is not None
            assert image is None and text is None and labels is None and mode is None and reduction is None
            assert label is None if generate else label is not None
        else:
            is_epalm = True
            assert perceptual_data is None and prompt is None and instruction is None and label is None
            assert image is not None and text is not None and mode in ['train', 'generate'] and return_dict
            perceptual_data = image
            prompt = text
            instruction = ['' for _ in range(len(text))]
            generate = (mode == 'generate')

        _ = self.feat_model(perceptual_data)
        if is_epalm:
            text.labels = labels
        else:
            text = self.tokenize_input(prompt, instruction, label)

        if generate:
            out_tokens = self.model_text.generate(
                input_ids=text.input_ids.to(self.accelerator.device),
                attention_mask=text.attention_mask.to(self.accelerator.device),
                **self.config.llm.generation.args,
                synced_gpus=self.config.distributed.fsdp,
                use_cache=self.config.llm.model.use_cache,
            )
            if is_epalm:
                return out_tokens
            generated_tokens = out_tokens[:,text.input_ids.shape[1]:]

            return GenerationModelOutput(
                input_tokens=text.input_ids,
                output_tokens=out_tokens,
                generated_tokens=generated_tokens,
            )
        else:
            return self.model_text(
                input_ids=text.input_ids,
                attention_mask=text.attention_mask,
                labels=text.labels,
                return_dict=True,
                loss_smothing=self.config.training.loss_smothing
            )

    def load(self, checkpoint_file_path, remove_in_proj=False):
        state_dict = torch.load(checkpoint_file_path, map_location=torch.device('cpu'))
        if remove_in_proj:
            del_keys = [key for key in state_dict if 'adapter_connectors.0.0' in key]
            assert len(del_keys) >= 1
            for key in del_keys:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)


class DepalmTrainer:
    def __init__(self, depalm_model, optimizer, scheduler, data_loaders: Dict, metrics: MetricEvaluator) -> None:
        unwraped_model = unwrap(depalm_model)
        self.config = unwraped_model.config
        self.accelerator = unwraped_model.accelerator
        self.tokenizer = unwraped_model.tokenizer

        self.depalm_model = depalm_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_loaders = data_loaders
        self.train_loader_cur_subsplit = None
        self.metrics = metrics

        self.log = self.accelerator.get_logger('DepalmTrainer')
        self.device = self.accelerator.device

        if self.config.compile:
            self.log.info("Compiling the model...")
            for batch in self.data_loaders['train']:
                with self.accelerator.autocast():
                    ans = self.depalm_model(
                        perceptual_data=batch.input_features.to(self.device).type(self.accelerator.input_dtype),
                        prompt=batch.input_text,
                        instruction=batch.instruction,
                        label=batch.get_single_output_text(),
                    )
                    ans.loss.backward()
                break
            self.log.info("Model compiled")

    def enumerate_data_epoch_subsplit(self, data_loader, cur_epoch):
        """ Cycle through dataset part when the dataset is split between multiple epochs """
        cur_dataset_split = cur_epoch % self.config.dataset.split_into_epochs
        if cur_dataset_split == 0:
            self.train_loader_cur_subsplit = enumerate(data_loader)
        for i_batch, batch in self.train_loader_cur_subsplit:
            yield i_batch, batch
            if int((i_batch+1) / len(data_loader) * self.config.dataset.split_into_epochs) > cur_dataset_split:
                break # Wait until next subsplit

    def train_single_epoch(self, data_loader, cur_epoch):
        self.log.info(f"train_single_epoch [epoch={cur_epoch}]")
        self.depalm_model.train()
        self.optimizer.zero_grad()

        log_freq = self.config.training.log_freq * self.config.training.accumulate_steps
        metric_logger = MetricLogger(delimiter="  ")
        for group in self.optimizer.param_groups:
            metric_logger.add_meter(f'lr_{group["name"]}', SmoothedValue(window_size=1, fmt='{value:.6f}'))

        self.log.info(f"Spliting data [epoch={cur_epoch}]")
        data_iter = metric_logger.log_every(self.accelerator, data_loader, log_freq, f'Train Epoch: [{cur_epoch}]')

        self.log.info(f"Start iterating batchs [epoch={cur_epoch}]")

        for i_batch, batch in enumerate(data_iter):
            cur_split_i_batch = i_batch

            current_step = cur_epoch // self.config.dataset.split_into_epochs + i_batch / len(data_loader)
            GlobalState.training_fract = current_step / self.config.training.epochs
            with self.accelerator.autocast():
                answer_output = self.depalm_model(
                    perceptual_data=batch.input_features.to(self.device).type(self.accelerator.input_dtype),
                    prompt=batch.input_text,
                    instruction=batch.instruction,
                    label=batch.get_single_output_text(),
                )
                loss = answer_output.loss
            loss_value = loss.item()

            if any([l.isnan() for l in self.accelerator.all_gather(loss.detach())]):
                raise ValueError("NaN loss")

            self.accelerator.backward(loss)
            self.scheduler.step(current_step)
            if (cur_split_i_batch + 1) % self.config.training.accumulate_steps == 0: # Gradient accumulation
                self.accelerator.step(self.depalm_model, self.optimizer, clip_grad=self.config.training.clip_norm)

            # Update printed metrics
            metric_logger.update(loss=loss_value)
            learning_rates = {f'lr_{group["name"]}': group["lr"] for group in self.optimizer.param_groups}
            metric_logger.update(**learning_rates)
            if self.accelerator.is_main_process and cur_split_i_batch % log_freq == 0:
                self.log.info(f"Learning rates: {learning_rates}")

        self.log.info(f"Averaged stats: {metric_logger.global_avg()}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    @torch.no_grad()
    def predict(self, data_loader, verbose=True, state_id='predict', max_frac=1):
        if len(data_loader) == 0:
            return [], []
        self.accelerator.barrier()

        verbose = verbose and self.accelerator.is_main_process
        self.depalm_model.eval()

        n_samples_from_data_loader = len(data_loader)

        if max_frac < 1:
            n_samples_from_data_loader = round(max_frac * n_samples_from_data_loader)
        out_references, out_predictions = [], []
        for i_batch, batch in enumerate(tqdm(data_loader, ncols=120, desc="Prediction", disable=not verbose, total=n_samples_from_data_loader)):
            with self.accelerator.autocast():
                model_out = self.depalm_model(
                    perceptual_data=batch.input_features.to(self.device).type(self.accelerator.input_dtype),
                    prompt=batch.input_text,
                    instruction=batch.instruction,
                    generate=True,
                )
            batch_rows = batch.unstack()

            for data_row, generated_tokens, output_tokens in zip(batch_rows, model_out.generated_tokens, model_out.output_tokens):
                response_raw = self.tokenizer.decode(output_tokens, skip_special_tokens=False)
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                response = response.strip()
                expected_answer = data_row.get_single_output_text()

                instruct = data_row.instruction.format(prompt=data_row.input_text)
                self.log.info(f'-')
                self.log.info(f'Question: {repr(data_row.input_text)}')
                self.log.info(f'Full instruction: {repr(instruct)}')
                self.log.info(f'Response (raw): {repr(response_raw)}')
                self.log.info(f'Response: {repr(response)}')
                self.log.info(f'Expected answer: {repr(expected_answer)}')

                if self.config.qualitative: # Dump qualitative result inside files
                    qual_dir = Path("qualitative")
                    qual_dir.mkdir(exist_ok=True)
                    example_id = len(out_references)
                    img_path = qual_dir / f'{example_id}.jpg'
                    text_path = qual_dir / f'{example_id}.txt'

                    img = data_row.original_features
                    if isinstance(img, Image.Image):
                        if self.config.dataset.resize_image_size:
                            img = img.resize((self.config.dataset.resize_image_size, self.config.dataset.resize_image_size))
                        img.save(str(img_path))
                    else:
                        print(f"Can't save features of type {type(data_row.original_features)}")

                    with text_path.open('w') as text_file:
                        text_file.write(f'Question: {repr(data_row.input_text)}\n')
                        text_file.write(f'Full instruction: \'{repr(instruct)}\'\n')
                        text_file.write(f'Response (raw): {repr(response_raw)}\n')
                        text_file.write(f'Response: {repr(response)}\n')
                        text_file.write(f'Expected answer: {repr(expected_answer)}\n')
                        if hasattr(data_row, 'output_text_list'):
                            text_file.write(f'\n\n')
                            for i, ans in enumerate(data_row.output_text_list):
                                text_file.write(f'Expected answer [{i}]: {repr(ans)}\n')

                out_references.append(data_row.detach_features())
                out_predictions.append(response)

            if i_batch >= n_samples_from_data_loader:
                break

        out_references = list(chain(*self.accelerator.all_gather_any(out_references)))
        out_predictions = list(chain(*self.accelerator.all_gather_any(out_predictions)))
        return out_references, out_predictions

    def evaluate(self, data_loader='test', state_id='evaluate', max_frac=1, save_as=None):
        if isinstance(data_loader, list):
            metric_dict = {}
            for d_loader in data_loader:
                metric_dict.update(self.evaluate(d_loader, state_id=state_id, max_frac=max_frac, save_as=save_as))
            self.log.info(f"All metrics: {metric_dict}")
            return metric_dict

        data_name = data_loader
        if isinstance(data_loader, str):
            data_loader = self.data_loaders[data_loader]
        else:
            data_name = data_loader.__class__.__name__
        if len(data_loader) == 0:
            return {}

        references, predictions = self.predict(data_loader, state_id=state_id, max_frac=max_frac)

        metric_dict = None
        if self.accelerator.is_main_process:
            metric_dict, info_dict = self.metrics.evaluate(references, predictions)
            metric_dict = {f'{data_name}_{metric_name}': metric_val for metric_name, metric_val in metric_dict.items()}

            self.log.info(f"Evaluate at {state_id} on data split {data_name}")
            self.log.info(f"Metrics: {metric_dict}")

        for d in self.accelerator.all_gather_any(metric_dict):
            if d is not None:
                metric_dict = d

        return metric_dict

    def save_state(self, state_id):
        # Save model
        self.accelerator.barrier()
        unwrapped_model = self.accelerator.unwrap_model(self.depalm_model)
        if not self.config.distributed.fsdp:
            if self.accelerator.is_main_process:
                state_dict = unwrapped_model.state_dict()
        else:
            offload_to_cpu = bool(self.accelerator.world_size > 1)
            save_policy = FullStateDictConfig(offload_to_cpu=offload_to_cpu, rank0_only=True)
            with FullyShardedDataParallel.state_dict_type(
                    self.depalm_model, StateDictType.FULL_STATE_DICT, save_policy
                ):
                state_dict = self.depalm_model.state_dict()

        if self.accelerator.is_main_process:
            SAVE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            for key in list(state_dict.keys()):
                try:
                    param = unwrapped_model.get_parameter(key)
                    if not param.requires_grad:
                        del state_dict[key]
                except AttributeError:
                    pass # Normalization layers, etc.

                for black_listed in SAVE_BLACKLIST:
                    clean_key = key.replace('.base_module', '').replace('._fsdp_wrapped_module', '')
                    if black_listed in key or black_listed in clean_key:
                        if key in state_dict:
                            del state_dict[key]
                        break

            torch.save(state_dict, SAVE_MODEL_PATH / f'model_checkpoint_{state_id}.pt')

        # Save training state
        self.log.warn("No trainining state save (Because of OutOfMemory errors with CUDA)")

    def train(self, train_on='train', eval_on='val', test_on='val'):
        times_training, times_evals = [], []
        total_start_time = time.time()

        self.accelerator.barrier()
        self.log.info("Start training")

        for epoch in range(self.config.training.start_epoch, self.config.training.epochs):
            self.log.info(f"Starting epoch {epoch}")
            time_start = time.time()
            _ = self.train_single_epoch(self.data_loaders[train_on], epoch)
            times_training.append(int(time.time()-time_start))
            self.log.info(f"Took {times_training[-1]//60}min for train step")

            self.save_state(f"epoch={epoch}")

            time_start = time.time()
            eval_results = self.evaluate(eval_on, state_id=f'eval_training_e={epoch}',
                max_frac=self.config.dataset.splits[eval_on].training_frac)
            times_evals.append(int(time.time()-time_start))
            self.log.info(f"Took {times_evals[-1]//60}min for training evaluation step")

        self.save_state("end_checkpoint")

        time_start = time.time()
        self.evaluate(test_on, state_id=f'eval_end_training_{self.config.training.epochs}_epochs', save_as='full_at_train_end')
        final_eval_time = int(time.time() - time_start)
        self.log.info(f"Took {final_eval_time//60}min for final evaluation step")

        total_time_str = str(datetime.timedelta(seconds=int(time.time() - total_start_time)))
        self.log.info(f'Total time of training loop ({self.config.training.epochs-self.config.training.start_epoch} epochs): {total_time_str}')

        time_stats = {
            'train_total': sum(times_training),
            'train_avg': sum(times_training) / len(times_training),
            'epochs': len(times_training),
            'eval_intermediate_total': sum(times_evals),
            'eval_intermediate_avg': sum(times_evals) / len(times_evals),
            'eval_total': sum(times_evals) + final_eval_time,
            'final_test': final_eval_time,
        }
        self.log.info(f'Time stats: {time_stats}')
