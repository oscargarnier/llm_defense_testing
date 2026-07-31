
MODEL_NAME = JosephusCheung/Guanaco

download_model:
	hf download $(MODEL_NAME)


TARGET_MODEL = llama2
LOG_FILE = data/AutoDAN/llama-2-7b-chat-hf_behaviors.json
ATTACK = AUTODAN
SAVE_SUFFIX = complete 
NIGHT_SUFFIX = complete

pair:
	python pair_testing.py | tee jboutput.txt

evaluate:
	python evaluate_defenses.py \
		--attack $(ATTACK) \
		--attack_logfile "AutoDAN/results/autodan_hga/vicuna_0_complete.json" \
		--max_new_tokens 512 \
		--save_suffix $(SAVE_SUFFIX) \
		--inference_batch_size 8 \
		--target_model vicuna \
		--device 1


confirm_determinism:
	python dictionary_utils.py \
		--reference_outputs AutoDAN/results/autodan_hga/vicuna_0_local.json \
		--new_outputs AutoDAN/results/autodan_hga/vicuna_0_normal.json \

autodan:
	python AutoDAN/autodan_eval.py \
		--attack_mode hga \
		--dataset_path data/advbench/harmful_behaviors.csv \
		--max_new_tokens 128 \
		--save_suffix $(SAVE_SUFFIX) \
		--model guanaco \

smooth_llm_evaluate:
	python evaluate_defenses.py \
		--attack $(ATTACK) \
		--attack_logfile "AutoDAN/results/autodan_hga/llama2_0_complete.json" \
		--defense SmoothLLM \
		--max_new_tokens 256 \
		--save_suffix swap7 \
		--inference_batch_size 1 \
		--smoothllm_pert_type RandomPatchPerturbation \
		--smoothllm_pert_pct 10 \
		--smoothllm_num_copies 7 \	
		--smoothllm_batch_size 7

nightrun:
	python AutoDAN/autodan_eval.py \
		--attack_mode hga \
		--dataset_path data/advbench/harmful_behaviors.csv \
		--max_new_tokens 128 \
		--save_suffix $(NIGHT_SUFFIX) \
		--model guanaco \

megadan:
	python AutoDAN/autodan_eval.py \
		--attack_mode hga \
		--max_new_tokens 128 \
		--save_suffix megadan \
		--continue_after_jailbroken \
		--dataset_path ./data/advbench/smaller_behaviors.csv

generate_behavior_files:
	python generate_behavior_files.py
