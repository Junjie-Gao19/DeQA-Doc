# DeQA-Doc: Adapting DeQA-Score to Document Image Quality Assessment

Junjie Gao, Runze Liu, Yingzhe Peng, Shujian Yang, Jin Zhang, Kai Yang and Zhiyuan You

[![paper](https://img.shields.io/badge/arXiv-Paper-green.svg)](https://arxiv.org/abs/2507.12796)
[![GitHub Stars](https://img.shields.io/github/stars/Junjie-Gao19/DeQA-Doc?style=social)](https://github.com/Junjie-Gao19/DeQA-Doc)

This repository is the implementation of DeQA-Doc: Adapting DeQA-Score to Document Image Quality Assessment training and inference code. We won first place in the **VQualA 2025 DIQA: Document Image Quality Assessment Challenge**.

## mPLUG-Owl2-7B Training 
### Installation
```bash
git clone https://github.com/Junjie-Gao19/DeQA-Doc.git
cd DeQA-Doc/DeQA-Score
pip install -e .
```
If you want to train, you need to install extra dependencies:
```bash
pip install -e .[train]
```
### Refer to the Readme of the DeQA repository
[DeQA](https://github.com/zhiyuanyou/DeQA-Score)

### Download pre-trained model
You can obtain the initial weight from [mPLUG-Owl2](https://huggingface.co/MAGAer13/mplug-owl2-llama2-7b)


[DIQA_model](https://www.modelscope.cn/models/zhalala/DeQA-Doc/summary) are different models trained separately in different dimensions

[DeQA-Mix](https://www.modelscope.cn/models/zhalala/DeQA-Doc-Mix/summary) is a separate model trained with multiple dimensions mixed

### Infer
```bash
sh scripts/infer.sh
```
When you finish infer, you need to use eval to transfer the result to the format of DIQA.
```bash
sh scripts/diqa_eval.sh
```
### Train
if you want to train your own model
```bash
sh scripts/train.sh 
or
sh scripts/train_lora.sh
```

## Qwen2.5-VL-7B Training
### Use Llamafactory framwork to train Qwen2.5-VL-7B model
#### Install Llamafactory
[Llamafactory](https://github.com/hiyouga/LLaMA-Factory)
#### Exchange src files
You need to exchange the files in Llamafactory with the files in this repository.
Their positions in Llama factory are consistent with those in this repository.
### Train
```bash
llamafactory-cli train examples/train_full/qwen2.5_vl_diqa_sft.yaml
```
### Infer
You should use the DeQA infer script to infer the result.
```bash
sh scripts/infer_qwen.sh
```

## Acknowledgements
This work is based on [DeQA-Score](https://github.com/zhiyuanyou/DeQA-Score). Sincerely thanks for this awesome work.

## Citation
If you find our work useful for your research and applications, please cite using the BibTeX:
```bash
@misc{gao2025deqadocadaptingdeqascoredocument,
      title={DeQA-Doc: Adapting DeQA-Score to Document Image Quality Assessment}, 
      author={Junjie Gao and Runze Liu and Yingzhe Peng and Shujian Yang and Jin Zhang and Kai Yang and Zhiyuan You},
      year={2025},
      eprint={2507.12796},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.12796}, 
}
```
