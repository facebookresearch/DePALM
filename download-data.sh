#!/usr/bin/env sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo "Downloading COCO"
mkdir -p depalm-data/COCO
wget -nc http://images.cocodataset.org/zips/train2014.zip -O depalm-data/COCO/train2014.zip
wget -nc http://images.cocodataset.org/zips/val2014.zip -O depalm-data/COCO/val2014.zip
wget -nc http://images.cocodataset.org/zips/test2014.zip -O depalm-data/COCO/test2014.zip

echo "\n\nExtracting COCO"
unzip -qn depalm-data/COCO/train2014.zip -d depalm-data/COCO && rm depalm-data/COCO/train2014.zip
unzip -qn depalm-data/COCO/val2014.zip -d depalm-data/COCO && rm depalm-data/COCO/val2014.zip
unzip -qn depalm-data/COCO/test2014.zip -d depalm-data/COCO && rm depalm-data/COCO/test2014.zip

echo "\n\nDownloading VQAv2"
mkdir -p depalm-data/vqa_v2
wget -nc https://nuage.isir.upmc.fr/index.php/s/ACRfZgaZTp9boZ8/download -O depalm-data/splits_epalm_json.zip
echo "Extracting VQAv2"
unzip -qn depalm-data/splits_epalm_json.zip -d depalm-data/splits_epalm_json
cp depalm-data/splits_epalm_json/data/karpathy_* depalm-data/vqa_v2/
cp depalm-data/splits_epalm_json/data/dataset_coco.json depalm-data/COCO
rm depalm-data/splits_epalm_json.zip

echo "\n\nDownloading OKVQA"
mkdir -p depalm-data/okvqa
wget -nc https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip -O depalm-data/okvqa/train2014_annotations.json.zip
wget -nc https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip -O depalm-data/okvqa/val2014_annotations.json.zip
unzip -qn depalm-data/okvqa/train2014_annotations.json.zip -d depalm-data/okvqa
unzip -qn depalm-data/okvqa/val2014_annotations.json.zip -d depalm-data/okvqa
rm depalm-data/okvqa/train2014_annotations.json.zip depalm-data/okvqa/val2014_annotations.json.zip
wget -nc https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip -O depalm-data/okvqa/train2014_questions.json.zip
wget -nc https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip -O depalm-data/okvqa/val2014_questions.json.zip
unzip -qn depalm-data/okvqa/train2014_questions.json.zip -d depalm-data/okvqa
unzip -qn depalm-data/okvqa/val2014_questions.json.zip -d depalm-data/okvqa
rm depalm-data/okvqa/train2014_questions.json.zip depalm-data/okvqa/val2014_questions.json.zip

echo "\n\nDownloading A-OKVQA"
mkdir -p depalm-data/aokvqa
wget -nc https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz -O depalm-data/aokvqa/aokvqa_v1p0.tar.gz
tar -xzvf depalm-data/aokvqa/aokvqa_v1p0.tar.gz --directory depalm-data/aokvqa
rm depalm-data/aokvqa/aokvqa_v1p0.tar.gz


echo "\n\nDownloading MSRVTT"
mkdir -p depalm-data/msrvtt
wget -nc https://www.robots.ox.ac.uk/~maxbain/frozen-in-time/data/MSRVTT.zip -O depalm-data/msrvtt/MSRVTT.zip
unzip -qn depalm-data/msrvtt/MSRVTT.zip -d depalm-data/msrvtt
mv depalm-data/msrvtt/MSRVTT/* depalm-data/msrvtt/
rm depalm-data/msrvtt/MSRVTT.zip
cp depalm-data/splits_epalm_json/data/msrvtt_caption_* depalm-data/msrvtt/

echo "\n\nDownloading AudioCaps"
echo "[WARNING: partial download only, as some data was removed from youtube]"
mkdir -p depalm-data/audiocaps
wget -nc https://github.com/cdjkim/audiocaps/raw/master/dataset/test.csv -O depalm-data/audiocaps/test.csv
wget -nc https://github.com/cdjkim/audiocaps/raw/master/dataset/train.csv -O depalm-data/audiocaps/train.csv
wget -nc https://github.com/cdjkim/audiocaps/raw/master/dataset/val.csv -O depalm-data/audiocaps/val.csv
python -c "__import__('audiocaps_download').Downloader(root_path='depalm-data/audiocaps/',n_jobs=16).download(format = 'wav')"

echo "\n\nDownloading timm models"
mkdir -p depalm-data/models/timm
wget -nc https://huggingface.co/timm/vit_base_patch16_224.augreg_in1k/resolve/main/pytorch_model.bin?download=true -O depalm-data/models/timm/vit_base_patch16_224.augreg_in1k.bin
wget -nc https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k/resolve/main/pytorch_model.bin?download=true -O depalm-data/models/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k.bin
wget -nc https://huggingface.co/timm/vit_base_patch16_224.augreg_in21k/resolve/main/pytorch_model.bin?download=true -O depalm-data/models/timm/vit_base_patch16_224.augreg_in21k.bin
wget -nc https://huggingface.co/timm/vit_large_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin?download=true -O depalm-data/models/timm/vit_large_patch16_224.augreg_in21k_ft_in1k.bin
wget -nc https://huggingface.co/timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k/resolve/main/open_clip_pytorch_model.bin?download=true -O depalm-data/models/timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k.bin

echo "\n\nDownloding TimeSformer"
wget -nc https://www.dropbox.com/s/g5t24we9gl5yk88/TimeSformer_divST_8x32_224_K400.pyth?dl=0 -O depalm-data/models/TimeSformer_divST_8x32_224_K400.pyth


echo "\n\nDownloding llama-7b"
python -c "__import__('transformers').AutoTokenizer.from_pretrained('luodian/llama-7b-hf', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"
python -c "__import__('transformers').AutoModel.from_pretrained('luodian/llama-7b-hf', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"

echo "\n\nDownloding llama2-7b (comment out this part if you don't have access on huggingface)"
python -c "__import__('transformers').AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"
python -c "__import__('transformers').AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"

echo "\n\nDownloding llama2-7b-chat (comment out this part if you don't have access on huggingface)"
python -c "__import__('transformers').AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"
python -c "__import__('transformers').AutoModel.from_pretrained('meta-llama/Llama-2-7b-chat-hf', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"

echo "\n\nDownloding vicuna"
python -c "__import__('transformers').AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"
python -c "__import__('transformers').AutoModel.from_pretrained('lmsys/vicuna-7b-v1.5', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"

echo "\n\nDownloding OPT-125M (use it for testing purposes)"
python -c "__import__('transformers').AutoTokenizer.from_pretrained('facebook/opt-125m', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"
python -c "__import__('transformers').AutoModel.from_pretrained('facebook/opt-125m', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"

echo "\n\nDownloding OPT-6.7b)"
python -c "__import__('transformers').AutoTokenizer.from_pretrained('facebook/opt-6.7b', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"
python -c "__import__('transformers').AutoModel.from_pretrained('facebook/opt-6.7b', use_auth_token=__import__('os').environ['HUGGINGFACE_TOKEN'],cache_dir='depalm-data/models')"


echo "\n\nDownloading evaluation modules"
aac-metrics-download --cache_path depalm-data

echo "\n\nFinished !"