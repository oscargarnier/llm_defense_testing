# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

# Generation and evaluation of LLaVA model on ScienceQA dataset

import argparse, json
from tqdm import tqdm

from datasets import load_dataset

import model_vlm

SYSTEM_PROMPT = ("A chat between a curious user and an artificial intelligence assistant. "
                 "The assistant gives helpful answers to the user's multiple-choice questions, "
                 "responding with the letter of the correct choice only.")

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="llava-hf/llava-1.5-7b-hf",
                    help="Name of the VLM model to load")
parser.add_argument("--max_new_tokens", type=int, default=64,
                    help="Max new tokens to generate")
parser.add_argument("--quick", type=int, default=None,
                    help="Quick test on few examples (default: full set)")
parser.add_argument("--output", type=str, default=None,
                    help="Path to output jsonlines file")
parser.add_argument("--custom", action="store_true", help="Use custom generation method")
model_vlm.add_custom_gen_args(parser) # Add RESTA-related args
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()
print(args)

ds = load_dataset("derek-thomas/ScienceQA")['test']
if args.verbose:
    print(len(ds), "examples in ScienceQA test set")
    num_no_image = sum(1 for example in ds if example['image'] is None)
    print(num_no_image, "examples have no image")

if args.quick is not None:
    ds = ds.select(range(args.quick))
    if args.verbose:
        print("Quick test on", len(ds), "examples")

model_vlm.load_model(args.model)

jsonl_file = None
if args.output:
    print(f"Writing output to {args.output}")
    jsonl_file = open(args.output, "w")

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

total_correct = 0
total_questions = 0
for example in tqdm(ds, bar_format="{n_fmt}/{total_fmt} | {elapsed}<{remaining} | {rate_fmt}"):
    question = example["question"]
    image = example["image"]
    has_image = image is not None
    choices = example["choices"]
    answer = LETTERS[example["answer"]]

    query = question + "\n"
    for idx, choice in enumerate(choices):
        query += f"{LETTERS[idx]}. {choice}\n"

    if args.verbose:
        print()
        print("Has image:", has_image)
        print(query)

    if args.custom:
        response = model_vlm.custom_generation(
            image, query,
            max_new_tokens=args.max_new_tokens,
            system_prompt=SYSTEM_PROMPT,
            noised_samples=args.noised_samples,
            noise_scale=args.noise_scale,
            noise_type=args.noise_type)
    else:
        response = model_vlm.generate_response(
            image, query,
            max_new_tokens=args.max_new_tokens,
            system_prompt=SYSTEM_PROMPT)
    correct = response.strip().upper().startswith(answer)
    if correct:
        total_correct += 1
    total_questions += 1

    if args.verbose:
        print("Response:", response)
        print("Ground truth:", answer)
        print("Correct!" if correct else "Incorrect.")
        print("Running accuracy: " f"{total_correct}/{total_questions} = {100*total_correct/total_questions:.2f}%")
        print()

    if jsonl_file is not None:
        result = {
            "question": question,
            "has_image": has_image,
            "choices": choices,
            "ground_truth": answer,
            "response": response,
            "correct": correct
        }
        jsonl_file.write(json.dumps(result) + "\n")
        jsonl_file.flush()

print("Final accuracy: " f"{total_correct}/{total_questions} = {100*total_correct/total_questions:.2f}%")
