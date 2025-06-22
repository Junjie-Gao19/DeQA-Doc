#!/bin/bash

# Qwen2.5-VL IQA评估脚本示例

# 设置模型路径
MODEL_PATH="/insdata2/jungao.jg/Qwen/merge/diqa_color_0621"

# 设置数据路径
META_PATHS=(
    "Data-DeQA-Score/DIQA/metas/test_diqa_500.json"
)

ROOT_DIR="/insdata2/chengyue.yk/datasets/DIQA/val/res/"

# 设置质量等级（根据你的模型训练时的设置调整）
LEVEL_NAMES=("excellent" "good" "fair" "poor" "bad")

# 设置输出目录
SAVE_DIR="results/qwen_color_0621/"

# 运行评估
python src/evaluate/iqa_eval_qwen_multi.py \
    --model-path $MODEL_PATH \
    --meta-paths "${META_PATHS[@]}" \
    --level-names "${LEVEL_NAMES[@]}" \
    --root-dir $ROOT_DIR \
    --save-dir $SAVE_DIR \
    --multi-gpu\