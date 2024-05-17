############################################### 
# Adapted from https://raw.githubusercontent.com/ylsung/VL_adapter/545fcbbdbbaec4c442de35567f6ae477ff4e8265/VL-T5/src/vqa_raw_data.py

from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
import json
import random

from omegaconf import OmegaConf
from PIL import Image

from .data_base import compute_cut, DataRowQASortedMO, FeaturesDataset, DataRowMultipleOutputs


@dataclass
class VQADataRow(DataRowMultipleOutputs):
    question_id: int = None
    question_type: str = None
    answer_type: str = None


class VQADataset(FeaturesDataset):
    def __init__(self, *args, 
            vqa_data_path,
            data_file,
            **kwargs
        ):
        self.data_file = data_file
        self.vqa_data_path = vqa_data_path
        super().__init__(*args, **kwargs)
    
    def _preload_dataset(self):
        data_file_path = Path(self.vqa_data_path) / self.data_file
        with data_file_path.open() as f:
            raw_data = json.load(f)
        
        all_data_rows = []
        for row in raw_data:
            img_file = f'{row["img_id"]}.jpg'
            _, img_dir, _ = row["img_id"].split('_')
            all_data_rows.append(VQADataRow(
                input_text=row['sent'],
                question_id=row['question_id'],
                features_path=str(Path(self.root) / img_dir / img_file),
                output_text=None,
                output_text_list=[ans['answer'] for ans in row['answers']],
                question_type=row['question_type'],
                answer_type=row['answer_type'],
            ))
        return all_data_rows

    def populate_data_row(self, data_row):
        data_row.input_features = Image.open(data_row.features_path).convert('RGB')
        super().populate_data_row(data_row)

    @classmethod
    def from_config(cls, config: OmegaConf, split, **kwargs):
        split_cfg = config.splits[split.value]
        return super().from_config(
            config,
            split=split,
            vqa_data_path=config.vqa_data_path,
            data_file=split_cfg.data_file,
            **kwargs
        )

class OkVqaDataset(FeaturesDataset):
    def _preload_dataset(self):
        if self.split.value == "test":
            return []
        questions_file = f'OpenEnded_mscoco_{self.split.value}2014_questions.json'
        answer_file = f'mscoco_{self.split.value}2014_annotations.json'
        coco_root = (Path(self.root) / '../COCO').resolve()

        annotations_data = json.load((Path(self.root) / answer_file).open())
        annotations = {ans['question_id']: ans for ans in annotations_data['annotations']}
        raw_data = json.load((Path(self.root) / questions_file).open())['questions']
        
        all_data_rows = []
        for row in raw_data:
            img_dir = f'{self.split.value}2014'
            img_file = f'COCO_{img_dir}_{row["image_id"]:012d}.jpg'
            cur_annotation = annotations[row["question_id"]]
            answers = [ans_str['raw_answer'] for ans_str in cur_annotation['answers']]
            
            img_path = str(coco_root / img_dir / img_file)
            assert Path(img_path).exists(), f"Image {img_path} doesn't exists"

            all_data_rows.append(VQADataRow(
                input_text=row['question'],
                question_id=row['question_id'],
                features_path=str(coco_root / img_dir / img_file),
                output_text=None,
                output_text_list=answers,
                question_type=annotations_data['question_types'][cur_annotation['question_type']],
                answer_type=cur_annotation['answer_type'],
            ))
        return all_data_rows

    def populate_data_row(self, data_row):
        data_row.input_features = Image.open(data_row.features_path).convert('RGB')
        super().populate_data_row(data_row)

class AOkVqaDataset(OkVqaDataset):
    @staticmethod
    def find_coco_img(coco_root, image_id):
        for split in ['train', 'val', 'test']:
            img_path = Path(coco_root) / f'{split}2014/COCO_{split}2014_{image_id:012d}.jpg'
            if img_path.exists():
                return str(img_path)
        raise FileNotFoundError(f"Can't find any COCO image with id {image_id:012d}")

    def _preload_dataset(self):
        if self.split.value == "test":
            return []
        data_file = f'aokvqa_v1p0_{self.split.value}.json'
        coco_root = (Path(self.root) / '../COCO').resolve()

        data_raw = json.load((Path(self.root) / data_file).open())
        
        all_data_rows = []
        for row in data_raw:
            all_data_rows.append(VQADataRow(
                input_text=row['question'],
                question_id=row['question_id'],
                features_path=self.find_coco_img(coco_root, row["image_id"]),
                output_text=None,
                output_text_list=row['direct_answers'],
                question_type='aokvqa',
                answer_type='aokvqa',
            ))
        return all_data_rows
