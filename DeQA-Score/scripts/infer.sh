export PYTHONPATH=./:$PYTHONPATH

python src/evaluate/iqa_eval_prompt.py \
	--level-names excellent good fair poor bad\
	--model-path  deqa_0618_color_norm_pair_1024 \
	--save-dir results/deqa_color_1024\
	--preprocessor-path ./preprocessor/ \
	--root-dir images_path \
	--meta-paths Data-DeQA-Score/DIQA/metas/test/test_diqa.json \
    --device cuda:0 \
    #--model-base MAGAer13__mplug-owl2-llama2-7b \
	#If you want to use Lora, add --model-base.