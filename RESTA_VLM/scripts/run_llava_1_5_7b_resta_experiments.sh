#!/usr/bin/env bash

# Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later

# Sequential RESTA experiment loops for LLaVA-1.5-7B.
# On a cluster, schedule the Python commands inside these loops as separate GPU jobs.

set -euo pipefail

noise_types=(normal hard)
scales=(0.001 0.002 0.005 0.01 0.015 0.02 0.025 0.03 0.04 0.05)
num_splits=10

usage() {
    cat <<'USAGE'
Usage: scripts/run_llava_1_5_7b_resta_experiments.sh [all|scienceqa|jailbreak|judge|asr]

Runs the sequential LLaVA-1.5-7B RESTA command loops from README.md.
For cluster runs, schedule the Python commands inside these loops as separate GPU jobs.
USAGE
}

run_scienceqa() {
    mkdir -p results

    for noise_type in "${noise_types[@]}"; do
        for scale in "${scales[@]}"; do
            python -u sqa_gen_eval.py \
                --model llava-hf/llava-1.5-7b-hf \
                --custom \
                --noised_samples 10 \
                --noise_type "$noise_type" \
                --noise_scale "$scale" \
                --output "results/llava-1.5-7b_ScienceQA_RESTA_${noise_type}_${scale}.jsonl"
        done
    done
}

run_jailbreak_generation() {
    local outdir

    for noise_type in "${noise_types[@]}"; do
        for scale in "${scales[@]}"; do
            outdir="results/llava-1.5-7b/JBV28K/${noise_type}"
            mkdir -p "$outdir"

            for ((split_index = 0; split_index < num_splits; split_index++)); do
                python -u generate_jailbreak.py \
                    --model llava-hf/llava-1.5-7b-hf \
                    --custom \
                    --noised_samples 10 \
                    --noise_type "$noise_type" \
                    --noise_scale "$scale" \
                    --split_index "$split_index" \
                    --num_splits "$num_splits" \
                    --output "${outdir}/scale_${scale}_index_${split_index}.jsonl" \
                    &> "${outdir}/scale_${scale}_index_${split_index}.log"
            done
        done
    done
}

run_jailbreak_judging() {
    local outdir

    for noise_type in "${noise_types[@]}"; do
        for scale in "${scales[@]}"; do
            outdir="results/llava-1.5-7b/JBV28K/${noise_type}"

            for ((split_index = 0; split_index < num_splits; split_index++)); do
                python -u jb_judge.py \
                    --response_only \
                    --input "${outdir}/scale_${scale}_index_${split_index}.jsonl" \
                    --output "${outdir}/scale_${scale}_index_${split_index}_rejudged.jsonl" \
                    &> "${outdir}/rejudge_scale_${scale}_index_${split_index}.log"
            done
        done
    done
}

run_asr() {
    local outdir

    for noise_type in "${noise_types[@]}"; do
        for scale in "${scales[@]}"; do
            outdir="results/llava-1.5-7b/JBV28K/${noise_type}"
            echo "ASR results for noise_type=${noise_type}, scale=${scale}"
            python asr_stats.py "${outdir}/scale_${scale}_index_*_rejudged.jsonl"
        done
    done
}

if [[ $# -gt 1 ]]; then
    usage >&2
    exit 2
fi

stage="${1:-all}"

case "$stage" in
    all)
        run_scienceqa
        run_jailbreak_generation
        run_jailbreak_judging
        run_asr
        ;;
    scienceqa)
        run_scienceqa
        ;;
    jailbreak | generate)
        run_jailbreak_generation
        ;;
    judge)
        run_jailbreak_judging
        ;;
    asr)
        run_asr
        ;;
    -h | --help)
        usage
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
