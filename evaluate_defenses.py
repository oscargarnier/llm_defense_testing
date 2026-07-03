import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "AutoDAN"))
import torch
import pandas as pd
import json
import random
import time
from tqdm.auto import tqdm
import argparse
import time
#import lib.perturbations as perturbations
#import lib.attacks as attacks
#import lib.language_models as language_models
#import lib.model_configs as model_configs
from defenses.factory import get_defense
from attacks.factory import get_attack
import numpy as np
import torch.nn as nn

from AutoDAN.utils.opt_utils import load_model_and_tokenizer
from AutoDAN.utils.string_utils import load_conversation_template
from AutoDAN.utils.eval_utils import check_for_attack_success, set_seed, update_gen_config
from AutoDAN.utils.references import MODEL_PATH_DICTS
from jailbreak_evaluators import SyntaxicEvaluator
from experiment_utils import (
    build_pending_prompts,
    build_results_path,
    get_processed_prompts,
    load_results_file,
    save_results_file,
)

def get_attack_data(attack_data_path):
    with open(attack_data_path, "r") as f:
        artifact_dataset = json.load(f)
    return artifact_dataset

def main(args):
    start_time = time.time()

    results_path = build_results_path(
        results_dir=args.results_dir,
        attack=args.attack,
        defense_type=args.defense,
        target_model=args.target_model,
        save_suffix=args.save_suffix,
    )
    results = load_results_file(results_path)
    results["__experiment_signature__"] = {
        "entry_type": "experiment_signature",
        "args": dict(vars(args)),
    }
    processed_prompts = get_processed_prompts(results)

    if not args.nosave:
        save_results_file(results_path, results)

    # Setup compute
    set_seed()
    device = f"cuda:{args.device}"

    # Setup conv template
    template_name = args.target_model
    conv_template = load_conversation_template(template_name)

    # Setup model
    model_path = os.path.expanduser(MODEL_PATH_DICTS[args.target_model])
    target_model , tokenizer = load_model_and_tokenizer(
        model_path,
        low_cpu_mem_usage=True,
        use_cache=False,
        device=device,
        quantize=args.quantize
    )
    print(f"Success: loaded model from path {model_path}")

    # Setup defense
    defense = get_defense(
        defense_type=args.defense,
        target_model=target_model,
        tokenizer=tokenizer,
        conv_template=conv_template,
        args=args
    )

    # Setup jailbreak evaluator
    jailbreak_evaluator = SyntaxicEvaluator()
    print(f"Using jailbreak evaluator: {jailbreak_evaluator.__class__.__name__}")

    # Setup attack dataset
    
    attack = get_attack(
        args.attack, 
        logfile = args.attack_logfile, 
        target_model = template_name,
        tokenizer = tokenizer, 
        conv_template = conv_template
    )

    num_jailbroken = 0

    start_time = time.time()
    artifact_start_time = start_time

    # Setup the generation config
    gen_config = update_gen_config(target_model.generation_config, args)

    pending_prompts = build_pending_prompts(attack.prompts, processed_prompts)
    num_jailbroken = sum(1 for result in results.values() if result.get("jailbroken"))

    # Process prompts in batches for batched inference
    batch_size = args.inference_batch_size
    for start in range(0, len(pending_prompts), batch_size):
        batch_items = pending_prompts[start:start + batch_size]
        batch_indices = [item_index for item_index, _ in batch_items]
        batch_prompts = [item_prompt for _, item_prompt in batch_items]

        if not batch_prompts:
            continue

        print(f"Evaluating artifacts {batch_indices[0]}..{batch_indices[-1]}...")

        batch_start_time = time.time()
        outputs = defense(batch_prompts, gen_config, batch_size=batch_size)
        batch_time = time.time() - batch_start_time

        # assign outputs back to individual artifacts
        per_item_time = batch_time / max(1, len(batch_prompts))
        for j, output in enumerate(outputs):
            i = batch_indices[j]
            prompt = batch_prompts[j]
            jailbroken = jailbreak_evaluator(output)
            if jailbroken:
                num_jailbroken += 1

            print(f"######################## INPUT ########################: \n {prompt.user_text_prompt}")
            print(f"######################## OUTPUT ########################: \n {output} \n\n  ######################## JAILBROKEN: {jailbroken} \n INFERENCE TIME: {per_item_time}s")

            result = {
                "goal": prompt.goal,
                "user_text_prompt": prompt.user_text_prompt,
                "output": output,
                "time": per_item_time,
                "jailbroken": jailbroken
            }

            results[str(i)] = result

            processed_prompts.add(prompt.user_text_prompt)

        if not args.nosave:
            save_results_file(results_path, results)

    print(f"Total inference time: {time.time() - start_time} seconds")
    print(f"Number of jailbroken artifacts: {num_jailbroken} out of {len(attack.prompts)}")


 
if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',type=str,default='./defense_testing_results')
    parser.add_argument( '--trial', type=int, default=0)

    # Targeted LLM
    parser.add_argument( '--target_model', type=str, default='llama2', choices=['vicuna', 'llama2'])

    # Experimental setup
    parser.add_argument( '--attack_logfile', type=str, default='data/GCG/vicuna_behaviors.json')
    parser.add_argument( '--attack', type=str, default='AUTODAN', choices=['AUTODAN', 'PAIR', 'GCG'])
    parser.add_argument( '--defense', type=str, default='NoDefense', choices=[ 'NoDefense', 'SmoothLLM' ])

    # SmoothLLM arguments
    parser.add_argument( "--smoothllm_batch_size", type=int, default=8, help="Number of prompts to run per generate call (batched inference).")
    parser.add_argument( '--smoothllm_num_copies', type=int, default=10,)
    parser.add_argument( '--smoothllm_pert_type', type=str, default='RandomSwapPerturbation', choices=[ 'RandomSwapPerturbation', 'RandomPatchPerturbation', 'RandomInsertPerturbation' ])
    parser.add_argument( '--smoothllm_pert_pct', type=int, default=10)

    # LLM technicals
    parser.add_argument( '--do_sample', action="store_true")
    parser.add_argument( '--quantize', action="store_true")
    parser.add_argument( "--max_new_tokens", type=int, default=64)
    parser.add_argument( "--inference_batch_size", type=int, default=8, help="Number of prompts to run per generate call (batched inference).")
    parser.add_argument( "--device", type=int, default=0)

    # XP context
    parser.add_argument( '--verbose', action="store_true")
    parser.add_argument( '--nosave', action="store_true")
    parser.add_argument("--save_suffix", type=str, default="")

    args = parser.parse_args()
    main(args)
