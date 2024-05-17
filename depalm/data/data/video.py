# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, List, Optional, Tuple, Literal
from pathlib import Path
import random
from collections import defaultdict
import json
import logging

import numpy as np
from torchvision import transforms
from PIL import Image
try:
    import decord
    from decord.bridge import to_torch
    from torchvision import transforms as transforms_lib
except ImportError:
    decord = None

from .data_base import FeaturesDataset, DataRowMultipleOutputs


def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ["rand", "middle"]:
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = [frame_indices[int(i)] for i in list(np.linspace(0, len(frame_indices)-1, num=max_num_frames))]
    else:
        raise ValueError
    return frame_indices

def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None, max_num_frames=-1):
    if decord is None:
        raise ModuleNotFoundError(f"{read_frames_decord} needs the package decord to be installed")
    video_reader = decord.VideoReader(video_path, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    frame_indices = get_frame_indices(
        num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = to_torch(frames)
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, frame_indices, duration


class MSRVTTCaptionDataset(FeaturesDataset):
    def __init__(self, *args, num_frames, sample_type, **kwargs):
        self.num_frames = num_frames
        self.sample_type = sample_type
        self.image_size = 224

        super().__init__(*args, auto_img_transforms=False, **kwargs)


    def _preload_dataset(self) -> List[DataRowMultipleOutputs]:
        data_dir = Path(self.root)

        self.features_label_transforms = None
        self.features_transform = None
        self.target_transform = None

        if self.sample_type == 'rand_middle':
            if self.split.value == "train":
                self.sample_type = "rand"
            else:
                self.sample_type = "middle"

        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

        if self.random_augment:
            self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(self.image_size,scale=(0.5, 1.0), interpolation=Image.BICUBIC),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(),
                    transformTypeTransform(),
                    normalize,
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size),interpolation=Image.BICUBIC),
                transformTypeTransform(),
                normalize,
            ])

        if self.split.value == "train":
            json_data = json.load((data_dir / 'msrvtt_caption_train7k.json').open('r'))
        elif self.split.value in ["test", "val"]:
            json_data = json.load((data_dir / 'msrvtt_caption_test.json').open('r'))
        else:
            assert False

        captions_per_video = defaultdict(lambda : [])
        for row in json_data:
            if isinstance(row['caption'], str):
                captions_per_video[row['video']].append(row['caption'])
            else:
                captions_per_video[row['video']].extend(row['caption'])

        return [
            DataRowMultipleOutputs(
                features_path=f'{data_dir}/videos/all/{video_id}',
                output_text_list=captions_per_video[video_id],
            ) for video_id in captions_per_video
        ]

    def populate_data_row(self, data_row: DataRowMultipleOutputs) -> None:
        frames, frame_indices, video_duration = read_frames_decord(
            data_row.features_path, self.num_frames, self.sample_type, max_num_frames=self.num_frames
        )
        frames = self.transform(frames) # Frames x Channels x H x W
        assert len(frames.shape) == 4
        frames = frames.transpose(0, 1)
        assert frames.shape[2:] == (224, 224) and frames.shape[0] == 3
        data_row.input_features = frames
        return super().populate_data_row(data_row)

    @classmethod
    def from_config(cls, config, split, **kwargs):
        return super().from_config(
            config,
            split,
            num_frames=config.num_frames,
            sample_type=config.sample_type,
            **kwargs
        )

class transformTypeTransform(Callable):
    def __call__(self, x):
        return x.float().div(255.0)
