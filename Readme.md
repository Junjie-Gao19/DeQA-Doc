# mPLUG-Owl2-7B Training 
## Installation
```bash
git clone https://github.com/Junjie-Gao19/DeQA-Doc.git
cd DeQA-Doc/DeQA-Score
pip install -e .
```
If you want to train, you need to install extra dependencies:
```bash
pip install -e .[train]
```
## Refer to the Readme of the DeQA repository
[DeQA](https://github.com/zhiyuanyou/DeQA-Score)

## Download pre-trained model
[mPLUG-Owl2](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b)

[DIQA_model](https://www.modelscope.cn/models/zhalala/DeQA-Doc/summary)

## Infer
```bash
sh scripts/infer.sh
```
When you finish infer, you need to use eval to transfer the result to the format of DIQA.
```bash
sh scripts/diqa_eval.sh
```
## Train
if you want to train your own model
```bash
sh scripts/train.sh 
or
sh scripts/train_lora.sh
```

# Qwen2.5-VL-7B Training
## Use Llamafactory framwork to train Qwen2.5-VL-7B model
### Install Llamafactory
[Llamafactory](https://github.com/hiyouga/LLaMA-Factory)
### Exchange src files
You need to exchange the files in Llamafactory with the files in this repository.
Their positions in Llama factory are consistent with those in this repository.
## Train
```bash
llamafactory-cli train examples/train_full/qwen2.5_vl_diqa_sft.yaml
```
## Infer
You should use the DeQA infer script to infer the result.
```bash
sh scripts/infer_qwen.sh
```