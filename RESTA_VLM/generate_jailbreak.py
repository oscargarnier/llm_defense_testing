# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

# Run generation with LLaVA model on JailbreakV-28K dataset

import argparse, json, os
from datetime import datetime
from tqdm import tqdm

import torch
from datasets import load_dataset

import model_vlm

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                    help="Name of the VLM model to load")
parser.add_argument("--max_new_tokens", type=int, default=1024,
                    help="Max new tokens to generate")
parser.add_argument("--output", type=str, default=None,
                    help="Path to output jsonlines file")
parser.add_argument("--split_index", type=int, default=0)
parser.add_argument("--num_splits", type=int, default=1)
parser.add_argument("--mini", action="store_true", help="Use mini split for quicker testing")
parser.add_argument("--custom", action="store_true", help="Use custom generation method")
model_vlm.add_custom_gen_args(parser) # Add RESTA-related args
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()
print(args)

start_time = datetime.now()
print(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

assert 0 <= args.split_index < args.num_splits, f"Invalid split_index={args.split_index} for num_splits={args.num_splits}"

model_vlm.load_model(args.model)

if args.mini: # Loads only 280 examples for quick testing
    ds = load_dataset("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K")['mini_JailBreakV_28K']
else: # Loads the full dataset
    ds = load_dataset("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K")['JailBreakV_28K']

if args.num_splits > 1:
    ds = ds.shard(num_shards=args.num_splits, index=args.split_index)
    if args.verbose:
        print(f"Using split {args.split_index} of {args.num_splits}, {len(ds)} examples")

existing_ids = set()
jsonl_file = None
if args.output:
    if os.path.exists(args.output):
        print(f"Resuming: reading existing indices from {args.output}")
        with open(args.output, "r") as f:
            for num, line in enumerate(f):
                try:
                    record = json.loads(line)
                    existing_ids.add(record['id'])
                except Exception as e:
                    print(f"Warning: failed to parse line {num + 1}: {e}")
        jsonl_file = open(args.output, "a")
    else:
        print(f"Writing output to {args.output}")
        jsonl_file = open(args.output, "w")

if args.custom:
    print("Using custom generation method")
    generate_func = lambda image, query, max_new_tokens: model_vlm.custom_generation(
        image, query, max_new_tokens=max_new_tokens,
        noised_samples=args.noised_samples,
        noise_scale=args.noise_scale,
        noise_type=args.noise_type
    )
else:
    print("Using model.generate()")
    generate_func = model_vlm.generate_response

for example in tqdm(ds, bar_format="{n_fmt}/{total_fmt} | {elapsed}<{remaining} | {rate_fmt}"):
    if example['id'] in existing_ids:
        if args.verbose:
            print(f"Skipping already processed id: {example['id']}")
        continue
    image_path = os.path.join("data", example['image_path'])
    jb_query = example['jailbreak_query']
    output = generate_func(image_path, jb_query, max_new_tokens=args.max_new_tokens)
    if args.verbose:
        print("#", example['id'], image_path)
        print("=== Malicious Objective ===")
        print(example['redteam_query'])
        print("=== Jailbreak Prompt ===")
        print(jb_query)
        print("=== Response ===")
        print(output)
        print()
    if jsonl_file:
        record = dict(example)
        record['response'] = output
        jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        jsonl_file.flush()

print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB")

# Add time stamp and elapsed time
end_time = datetime.now()
print(f"Finished at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Elapsed time: {end_time - start_time}")

if jsonl_file:
    jsonl_file.close()
