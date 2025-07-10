#!/bin/bash
export PYTHONPATH=./:$PYTHONPATH

LOAD="modelpath"

deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 6688 src/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $LOAD \
    --version v1 \
    --dataset_type pair \
    --level_prefix "quality of the image is" \
    --level_names excellent good fair poor bad \
    --softkl_loss True \
    --weight_rank 1.0 \
    --weight_softkl 1.0 \
    --weight_next_token 0.005 \
    --continuous_rating_loss True \
    --closeset_rating_loss True \
    --use_fix_std True \
    --detach_pred_std True \
    --data_paths /Data-DeQA-Score/DIQA/metas/train_diqa_pair_sharpness.json \
                /Data-DeQA-Score/KONIQ/metas/mos.json \
    --data_weights 1 1 \
    --image_folder image_path\
    --output_dir output_path \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs 3 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "no" \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --tune_visual_abstractor True \
    --freeze_vision_model False \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to tensorboard
