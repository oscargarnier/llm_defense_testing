<!--
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
-->

# Directional Embedding Smoothing for Robust Vision Language Models

This repository provides the experimental code for our paper
["Directional Embedding Smoothing for Robust Vision Language Models"](https://arxiv.org/abs/2603.15259)
by Ye Wang, Jing Liu, Toshiaki Koike-Akino.

These experiments investigate robust VLMs, via an inference-time defense against multi-modal jailbreak attacks.
This defense extends the Randomized Embedding Smoothing and Token Aggregation (RESTA) defense, that we developed in our earlier paper
["Smoothed Embeddings for Robust Language Models"](https://arxiv.org/abs/2501.16497) by Ryo Hase, Md Rafi Ur Rashid, Ashley Lewis, Jing Liu, Toshiaki Koike-Akino, Kieran Parsons, Ye Wang.

However, note that this repository only provides experiments for our more recent robust VLM paper.
We reimplemented RESTA from scratch in this repo, while generalizing to support VLMs.


## Environment Setup

### Requirements and model access

The code assumes a CUDA GPU with enough memory for the selected VLM and judge model.
A 48 GB GPU is sufficient for the paper experiments.

Install the required packages with:

```sh
pip install -r requirements.txt
```

For customizing installation for specific CUDA versions, please see PyTorch installation instructions:
https://pytorch.org/get-started/locally/

Running this code will automatically download several models from Hugging Face.
In particular, the core paper experiments use the following models:

- https://huggingface.co/llava-hf/llava-1.5-7b-hf
- https://huggingface.co/google/gemma-3-4b-it
- https://huggingface.co/meta-llama/Llama-Guard-3-8B

Before using any gated models, you need to first accept any required model terms on Hugging Face, and request access.
Then, login from the machine that will run the jobs:

```sh
hf auth login
```

### Data and output layout

Create local data and results directories before running experiments:

```sh
mkdir -p data results
```

ScienceQA is loaded automatically through Hugging Face datasets from:
https://huggingface.co/datasets/derek-thomas/ScienceQA

JailbreakV-28K is only partially loaded from Hugging Face:
https://huggingface.co/datasets/JailbreakV-28K/JailBreakV-28k

The Hugging Face dataset provides the CSV metadata and only a fraction of the image files.
The remaining JailbreakV-28K images must be manually downloaded after approval via the Google form linked from the dataset page.
Place the downloaded folders under the `data/` directory so these paths exist:

```text
data/figstep
data/llm_transfer_attack
data/query_related
```

The generation script resolves each JailbreakV image as `data/<image_path from dataset metadata>`.


## Usage

The commands below are written as plain `python` invocations.
Use these as the canonical workflow, then wrap them with `srun`, `sbatch`, `tmux`, or another scheduler as needed.

### Quick smoke tests

Run a small ScienceQA evaluation to check model loading, image/text preprocessing, and generation:

```sh
python -u sqa_gen_eval.py --quick 100 \
    --output results/smoke_ScienceQA.jsonl
```

Run the 280-example mini split of JailbreakV-28K to check that the manually downloaded images are in the expected location:

```sh
python -u generate_jailbreak.py --mini \
    --output results/smoke_JBV28K.jsonl
```

For debugging outputs, add `--verbose` to either script.

### Baseline ScienceQA evaluation

Evaluate a VLM on the ScienceQA test split:

```sh
python -u sqa_gen_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --output results/llava-1.5-7b_ScienceQA.jsonl
```

Reference results:

- `llava-hf/llava-1.5-7b-hf`: 2717 / 4241 = 64.07%
- `llava-hf/llava-1.5-13b-hf`: 2923 / 4241 = 68.92%

ScienceQA outputs include a `correct` field, and the script prints final accuracy when it finishes.

### Baseline JailbreakV-28K generation

Generate responses for all JailbreakV-28K attacks:

```sh
python -u generate_jailbreak.py \
    --model llava-hf/llava-1.5-7b-hf \
    --output results/llava-1.5-7b_JBV28K.jsonl
```

Long JailbreakV-28K runs can be split into shards with `--num_splits` and `--split_index`.
For example, this runs four shards sequentially; on a cluster, schedule each shard as a separate GPU job:

```sh
for index in 0 1 2 3; do
    python -u generate_jailbreak.py \
        --model llava-hf/llava-1.5-7b-hf \
        --split_index "$index" \
        --num_splits 4 \
        --output "results/llava-1.5-7b_JBV28K_split_${index}.jsonl"
done
```

The generation script resumes an existing output file by reading already processed example IDs and appending only missing records.

### Judging jailbreak outputs

Use `jb_judge.py` to label generated responses with Llama Guard, with response-only judging; this avoids counting unsafe content in the attack prompt itself as a model failure.

```sh
python -u jb_judge.py \
    --input results/llava-1.5-7b_JBV28K.jsonl \
    --output results/llava-1.5-7b_JBV28K_rejudged.jsonl \
    --response_only
```

The default judge model is `meta-llama/Llama-Guard-3-8B`.
To reproduce older JailbreakV-style comparisons, use the older Llama Guard 2 model:

```sh
python -u jb_judge.py \
    --input results/llava-1.5-7b_JBV28K.jsonl \
    --output results/llava-1.5-7b_JBV28K_judged2.jsonl \
    --model meta-llama/Meta-Llama-Guard-2-8B \
    --response_only
```

To judge the full conversation context instead of only the model response, omit `--response_only`.
This is useful as an ablation, but it can introduce false positives, when the prompt itself contains unsafe content, and especially if the model has refused with an empty response (which should be considered safe).
Note: this is an issue with the judge model getting confused and judging the safety of the prompt instead of the response, despite instructions to judge only the response.

### ASR summaries

Compute attack success rate (ASR) summaries from judged JailbreakV-28K outputs:

```sh
python asr_stats.py results/llava-1.5-7b_JBV28K_rejudged.jsonl
python asr_stats.py "results/llava-1.5-7b_JBV28K_split_*_rejudged.jsonl"
```

`asr_stats.py` accepts one or more paths or glob patterns.
It reports overall safe/unsafe/other counts, ASR by JailbreakV format, and ASR by `llm_transfer_attack` image type.

### RESTA experiments

RESTA is enabled through the custom generation path:

- `--custom`: use the custom autoregressive generation loop in [model_vlm.py](./model_vlm.py)
- `--noised_samples`: number of noised samples to aggregate; our RESTA experiments mainly used `10`
- `--noise_type`: `normal` for isotropic embedding noise or `hard` for directional noise
- `--noise_scale`: scalar noise level

The default value of `--noised_samples 0` turns off RESTA and falls back to plain greedy decoding via our custom generation loop.

Run a single RESTA ScienceQA utility evaluation:

```sh
python -u sqa_gen_eval.py \
    --model llava-hf/llava-1.5-7b-hf \
    --custom \
    --noised_samples 10 \
    --noise_type normal \
    --noise_scale 0.01 \
    --output results/llava-1.5-7b_ScienceQA_RESTA_normal_0.01.jsonl
```

Run a single RESTA JailbreakV-28K safety generation shard:

```sh
python -u generate_jailbreak.py \
    --model llava-hf/llava-1.5-7b-hf \
    --custom \
    --noised_samples 10 \
    --noise_type hard \
    --noise_scale 0.01 \
    --split_index 0 \
    --num_splits 10 \
    --output results/llava-1.5-7b_JBV28K_RESTA_hard_0.01_split_0.jsonl
```

Then judge and summarize the RESTA output as usual:

```sh
python -u jb_judge.py \
    --input results/llava-1.5-7b_JBV28K_RESTA_hard_0.01_split_0.jsonl \
    --output results/llava-1.5-7b_JBV28K_RESTA_hard_0.01_split_0_rejudged.jsonl \
    --response_only

python asr_stats.py "results/llava-1.5-7b_JBV28K_RESTA_hard_0.01_split_*_rejudged.jsonl"
```

The scripts below keep the full sweeps as sequential examples to illustrate data handling and command flow; for actual experiments, you can schedule the Python commands inside them as separate GPU jobs run in parallel.
See the later section on "Adapting commands to Slurm".

#### LLaVA-1.5-7B RESTA experiments

The LLaVA RESTA sweep commands are collected in [scripts/run_llava_1_5_7b_resta_experiments.sh](./scripts/run_llava_1_5_7b_resta_experiments.sh).
The script includes ScienceQA utility evaluation, JailbreakV-28K generation over 10 shards, response-only judging, and ASR aggregation.

```sh
bash scripts/run_llava_1_5_7b_resta_experiments.sh
```

To run one stage at a time, pass `scienceqa`, `jailbreak`, `judge`, or `asr`:

```sh
bash scripts/run_llava_1_5_7b_resta_experiments.sh scienceqa
```

#### Gemma-3-4B RESTA experiments

The Gemma RESTA sweep commands are collected in [scripts/run_gemma_3_4b_resta_experiments.sh](./scripts/run_gemma_3_4b_resta_experiments.sh).
The script uses the Gemma model flag and RESTA grid, splits JailbreakV-28K generation over 20 shards, judges the generated shards, and aggregates ASR.

```sh
bash scripts/run_gemma_3_4b_resta_experiments.sh
```

To run one stage at a time, pass `scienceqa`, `jailbreak`, `judge`, or `asr`:

```sh
bash scripts/run_gemma_3_4b_resta_experiments.sh scienceqa
```

### Adapting commands to Slurm

The scripts and examples above use plain Python commands run sequentially.
On a Slurm cluster, wrap each command with local resource settings:

```sh
srun --unbuffered \
    --partition YOUR_GPU_PARTITION \
    --gpus 1 \
    --time 300 \
    python -u sqa_gen_eval.py \
        --model llava-hf/llava-1.5-7b-hf \
        --output results/llava-1.5-7b_ScienceQA.jsonl
```

For JailbreakV-28K generation, you may prefer to use Slurm arrays to limit and manage concurrent jobs.
The task ID can map to a model, noise type, scale, and split index; each array task should write one output file and one log file.

Minimal array skeleton:

```sh
#!/bin/bash
#SBATCH --partition=YOUR_GPU_PARTITION
#SBATCH --gpus=1
#SBATCH --time=3600
#SBATCH --array=0-99%20

scales=(0.001 0.002 0.005 0.01 0.015 0.02 0.025 0.03 0.04 0.05)
num_splits=10

task_id=${SLURM_ARRAY_TASK_ID}
scale_idx=$(( task_id / num_splits ))
split_idx=$(( task_id % num_splits ))
scale=${scales[$scale_idx]}

mkdir -p results

python -u generate_jailbreak.py \
    --custom \
    --noised_samples 10 \
    --noise_type hard \
    --noise_scale "$scale" \
    --split_index "$split_idx" \
    --num_splits "$num_splits" \
    --output "results/jbv_hard_scale_${scale}_split_${split_idx}.jsonl" \
    &> "results/jbv_hard_scale_${scale}_split_${split_idx}.log"
```

Replace partition, time, concurrency, and node constraints with values appropriate for the cluster.

## Paper PDF Links

Our related papers are available on arXiv:

- https://arxiv.org/abs/2603.15259
- https://arxiv.org/abs/2501.16497

Or as MERL technical reports:

- https://www.merl.com/publications/TR2026-049
- https://www.merl.com/publications/TR2024-170


## Citation

If you use the software, please cite the following:

```bibtex
@inproceedings{wang2026directional,
  title = {Directional Embedding Smoothing for Robust Vision Language Models},
  author = {Ye Wang and Jing Liu and Toshiaki Koike-Akino},
  booktitle = {International Conference on Learning Representations (ICLR) Workshop on Agents in the Wild},
  year = {2026},
  month = apr,
  url = {https://arxiv.org/abs/2603.15259},
}

@inproceedings{hase2024smoothed,
  title = {Smoothed Embeddings for Robust Language Models},
  author = {Ryo Hase and Md Rafi Ur Rashid and Ashley Lewis and Jing Liu and Toshiaki Koike-Akino and Kieran Parsons and Ye Wang},
  booktitle = {Safe Generative AI Workshop at Neural Information Processing Systems (NeurIPS)},
  year = {2024},
  month = dec,
  url = {https://arxiv.org/abs/2501.16497},
}
```

## Contact

Ye Wang <yewang@merl.com>

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for our policy on contributions.

## License

Released under `AGPL-3.0-or-later` license, as found in the [LICENSE.md](LICENSE.md) file.

All files, except as noted below:

```text
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)

SPDX-License-Identifier: AGPL-3.0-or-later
```

Portions of `jb_judge.py` (evaluation prompts) are adapted from JailbreakV-28K:

```text
Copyright (C) 2026 Mitsubishi Electric Research Laboratories (MERL)
Copyright (C) 2024 Weidi Luo, Siyuan Ma, Xiaogeng Liu, Xiaoyu Guo, Chaowei Xiao

SPDX-License-Identifier: AGPL-3.0-or-later
SPDX-License-Identifier: MIT
```
