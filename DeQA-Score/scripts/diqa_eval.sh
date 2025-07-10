export PYTHONPATH=./:$PYTHONPATH

python src/evaluate/cal_plcc_srcc.py \
	--level_names excellent good fair poor bad\
	--pred_paths results/deqa_color_1024/test_diqa.json\
    --save_path  results/deqa_color_1024/result.txt\
	--gt_paths Data-DeQA-Score/DIQA/metas/test/test_diqa.json\
