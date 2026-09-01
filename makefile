## Useful commands for running the AutoDAN evaluation framework


MODEL_NAME = JosephusCheung/Guanaco

# No internet access when within a process container,
# So the model needs to be downloaded first
download_model:
	hf download $(MODEL_NAME)


TARGET_MODEL = llama2
LOG_FILE = data/AutoDAN/llama-2-7b-chat-hf_behaviors.json
ATTACK = AUTODAN
SAVE_SUFFIX = complete 
NIGHT_SUFFIX = complete

pair:
	python pair_testing.py | tee jboutput.txt

>>>>>>> 4dbf7fd14495a8bd05bdd23a714fecaf33713b30
evaluate:
	python evaluate_defenses.py \
		--attack $(ATTACK) \
		--attack_logfile "AutoDAN/results/autodan_hga/guanaco_0_complete.json" \
		--max_new_tokens 512 \
		--save_suffix $(SAVE_SUFFIX) \
		--inference_batch_size 8 \
		--target_model guanaco \
		--device 1

# This is used to compare two output files
# Those of the same model but autodan vs inference
confirm_determinism:
	python dictionary_utils.py \
		--reference_outputs AutoDAN/results/autodan_hga/vicuna_0_local.json \
		--new_outputs AutoDAN/results/autodan_hga/vicuna_0_normal.json \

# Example of autodan attack
autodan:
	python AutoDAN/autodan_eval.py \
		--attack_mode hga \
		--dataset_path data/advbench/harmful_behaviors.csv \
		--max_new_tokens 128 \
		--save_suffix $(SAVE_SUFFIX) \
		--model guanaco \

# This is an example of the full pipeline
# You can specify the attack artifacts that you want
# The defense you want to apply
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

## This is used for the seperate nightrun evaluation
NIGHT_SUFFIX = complete

nightrun:
	python AutoDAN/autodan_eval.py \
		--attack_mode hga \
		--dataset_path data/advbench/harmful_behaviors.csv \
		--max_new_tokens 128 \
		--save_suffix $(NIGHT_SUFFIX) \
		--model guanaco \

## Testing the the autodan attack for 100 iterations regardless
megadan:
	python AutoDAN/autodan_eval.py \
		--attack_mode hga \
		--max_new_tokens 128 \
		--save_suffix megadan \
		--continue_after_jailbroken \
		--dataset_path ./data/advbench/smaller_behaviors.csv

generate_behavior_files:
	python generate_behavior_files.py
