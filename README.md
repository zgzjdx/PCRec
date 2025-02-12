# UNBERT

The official pytorch implementation for our paper “PCRec: A Multi-Interest News Recommendation Framework with
Prompt-Guided Cross-view Contrastive Learning”.

## Requirements
```bash
see requirements.txt
```

This code requires a GPU setup with total memory greater than 90GB. 
For example, you can use 4 NVIDIA 3090/4090/V100 GPUs to run the training and evaluation.
Ensure your setup meets this requirement to avoid out-of-memory issues.

## Data preparation
For the MIND dataset, please download at https://msnews.github.io

## Usage
Before running the code, download the bert-base-uncased model from https://huggingface.co/google-bert/bert-base-uncased 
and place it in the pretrainedModel folder.


```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --split small --batch_size 128 --lr 2e-5 --use_amp --eval
```
For more parameter settings, please refer to `run.py`.

## Acknowledgment
https://github.com/reczoo/RecZoo/tree/main/pretraining/news/UNBERT
