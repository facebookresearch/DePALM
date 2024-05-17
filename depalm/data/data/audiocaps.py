# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, List, Optional, Tuple, Literal
from pathlib import Path
import logging
import random
import csv
from collections import defaultdict

from omegaconf import OmegaConf
import numpy as np
import torch
import torchaudio
import librosa

from .data_base import FeaturesDataset, compute_cut, DataRowMultipleOutputs


class AudioCapsDataset(FeaturesDataset):
    AUDIO_MAP = {}

    def __init__(
        self,
        split: "FeaturesDataset.Split",
        root: str,
        data_paths: List[str],
        transform: Optional[Callable] = None,
        max_rows: Optional[int] = None,
        random_augment: bool = False,
        noise: bool = False,
        skip_norm: bool = False,
        normalize_text: bool = False,
        logger = None,
        instructions: Optional[List[str]] = None,
        instruction_mode: Literal[None, False, 'no', 'default', 'random', 'first'] = None,
        audio_model = None,
        split_multirow_train=False,
    ) -> None:
        self.root = root
        self.data_paths = data_paths
        self.split = split

        self.features_transform = transform
        self.target_transform = None
        self.features_label_transforms = None
        self.normalize_text = normalize_text
        self.split_multirow_train = split_multirow_train

        self.max_rows = max_rows
        self.instructions = instructions or []
        self.instruction_mode = instruction_mode
        self.audio_model = audio_model

        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
        self.logger = logger

        # audio args
        self.melbins = 128
        self.target_length = 1024
        self.freqm_p = 24
        self.timem_p = 96

        self.norm_mean = -4.2677393
        self.norm_std = 4.5689974
        self.skip_norm = skip_norm
        self.noise = noise
        self.random_augment = random_augment

        self.freqm = torchaudio.transforms.FrequencyMasking(self.freqm_p)
        self.timem = torchaudio.transforms.TimeMasking(self.timem_p)

        # Pre-load dataset
        self.data = self._preload_dataset()
        self._split_multirows()
        self.logger.info(f"Loaded {len(self)} rows from data split {self.split.value}")
        self.data = self._limit_dataset_size(self.data)

    def _preload_dataset(self) -> List[DataRowMultipleOutputs]:
        data_info_path = Path(self.root) / f'{self.split.value}.csv'
        with open(data_info_path) as csvfile:
            datarows = list(csv.reader(csvfile))
        datarows = datarows[1:] # Remove header: ['audiocap_id', 'youtube_id', 'start_time', 'caption']

        if not self.AUDIO_MAP:
            for data_path in self.data_paths:
                data_path = data_path
                for audio_path in Path(data_path).iterdir():
                    self.AUDIO_MAP[audio_path.name.split(".")[0]] = str(audio_path) # if audiocaps id
                    self.AUDIO_MAP[audio_path.name[:11]] = str(audio_path) # if yt-id

        data = defaultdict(lambda : [])
        not_found = 0
        for audiocap_id, yt_id, _, caption in datarows:
            if self.normalize_text:
                caption = caption.replace(',', '').replace('"', '')
            if audiocap_id in self.AUDIO_MAP:
                data[audiocap_id].append(caption)
            elif yt_id in self.AUDIO_MAP:
                data[yt_id].append(caption)
            else:
                not_found += 1

        data = [
            DataRowMultipleOutputs(
                features_path=self.AUDIO_MAP[audiocap_id],
                output_text_list=all_captions,
            )
            for audiocap_id, all_captions in data.items()
        ]

        self.logger.info(f"{not_found} audio files not found for split {self.split.value}")
        random.shuffle(data)
        return data

    def populate_data_row(self, data_row) -> None:
        if self.audio_model == 'audio_features':
            waveform, sr = torchaudio.load(data_row.features_path)
            waveform = waveform - waveform.mean()
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                        window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                        frame_shift=10)

            n_frames = fbank.shape[0]
            p = self.target_length - n_frames

            # cut and pad
            if p > 0:
                m = torch.nn.ZeroPad2d((0, 0, 0, p))
                fbank = m(fbank)
            elif p < 0:
                fbank = fbank[0:self.target_length, :]

            if self.random_augment: # Data Augmentation
                fbank = torch.transpose(fbank, 0, 1)
                # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
                fbank = fbank.unsqueeze(0)

                if self.freqm_p != 0:
                    fbank = self.freqm(fbank)
                if self.timem_p != 0:
                    fbank = self.timem(fbank)

                # squeeze it back, it is just a trick to satisfy new torchaudio version
                fbank = fbank.squeeze(0)
                fbank = torch.transpose(fbank, 0, 1)

            if not self.skip_norm: # normalize the input
                fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
            else: # skip normalization the input if you are trying to get the normalization stats.
                pass

            if self.random_augment and self.noise == True:
                fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
                fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)
            audio_data = fbank  # torch.Size([1024, 128])
        elif self.audio_model == 'raw_audio':
            audio_data, _ = librosa.load(data_row.features_path, sr=48000) # sample rate should be 48000
            N = 480000
            while len(audio_data) < N:
                audio_data = audio_data.repeat(max(2, (len(audio_data)+N-1)//N))
            audio_data = audio_data[:N]
            assert audio_data.shape == (N, )
            audio_data = torch.tensor(audio_data)
        else:
            raise ValueError(f"Can't load audio in format for model {self.audio_model}")

        data_row.input_features = audio_data
        super().populate_data_row(data_row)

    @classmethod
    def from_config(cls, config: OmegaConf, split, **kwargs):
        return cls(
            split=split,
            data_paths=config.data_paths,
            random_augment=config.splits[split.value].random_augment,
            instructions=config.instructions,
            instruction_mode=config.insert_instruction,
            noise=config.noise,
            skip_norm=config.skip_norm,
            normalize_text=config.normalize_text,
            audio_model=config.audio_model_type,
            split_multirow_train=config.split_multirow_train,
            **kwargs
        )