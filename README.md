# AVR: Learning Adaptive Reasoning Paths for Efficient Visual Reasoning

This repository contains the training code for AVR, an adaptive visual reasoning framework for reducing overthinking in visual reasoning models. AVR decomposes visual reasoning into three cognitive functions, visual perception, logical reasoning, and answer application, and trains the model to choose among direct-answer, perception-only, and full-format responses based on task difficulty.

AVR targets reasoning path redundancy in vision-language reasoning. The goal is to preserve answer correctness while reducing unnecessary token usage through adaptive format selection. In the paper, AVR reduces token usage by 50-90% across benchmarks while maintaining competitive accuracy.

## Highlights

- Identifies reasoning path redundancy in visual reasoning models
- Uses multi-format SFT to teach structured, task-adaptive outputs
- Uses FS-GRPO to optimize correctness and efficiency jointly
- Supports direct-answer, perception-only, and full reasoning responses in one model

## Overview

AVR uses a two-stage training pipeline:

1. Supervised fine-tuning teaches the model to follow the structured output formats used by AVR.
2. FS-GRPO further optimizes format selection so the model prefers the most efficient response that still remains correct.

The reward used in FS-GRPO combines:

- answer correctness
- format-dependent reward shaping
- response-length regularization
- intra-group diversity bonus for encouraging exploration early in training

## Repository Structure

- [reward_functions](reward_functions): adaptive reward used for FS-GRPO training
- [train_configs](train_configs): SFT and FS-GRPO configuration files
- [setup_fs-grpo_env.sh](setup_fs-grpo_env.sh): environment setup script for verl-based training
- [train-sft.sh](train-sft.sh): entry script for SFT with LLaMA-Factory
- [train-fs-grpo.sh](train-fs-grpo.sh): entry script for FS-GRPO training with `verl`
- [merge_fs-grpo_ckpt.sh](merge_fs-grpo_ckpt.sh): checkpoint merge helper
- [LLaMA-Factory](LLaMA-Factory): SFT training framework submodule
- [verl](verl): RL training framework submodule

## Output Formats

AVR trains the model to select one of three response formats:

1. Direct answer
```text
<answer>...</answer>
```

2. Perception plus answer
```text
<perception>...</perception>
<answer>...</answer>
```

3. Full reasoning
```text
<perception>...</perception>
<reasoning>...</reasoning>
<answer>...</answer>
```

These formats correspond to progressively richer reasoning paths. During training, the model is encouraged to use the shortest format that can still solve the task correctly.

## SFT Training

SFT is run through LLaMA-Factory and teaches the base VLM to emit AVR-style structured responses.

Example entrypoint:

```bash
bash train-sft.sh
```

The current script launches:

```bash
FORCE_TORCHRUN=1 llamafactory-cli train ../train_configs/qwen3vl_2b_full_sft_all.yaml
```

The released repository keeps a single SFT config in [train_configs](train_configs):

- `qwen3vl_2b_full_sft_all.yaml`

Shared SFT characteristics:

- full-parameter fine-tuning
- frozen vision tower and multimodal projector
- one training epoch in the checked-in configs

Before running SFT, make sure:

- the LLaMA-Factory submodule is initialized
- the target conda environment is available
- the training dataset referenced in the YAML config is registered in LLaMA-Factory

## Environment Setup

Use the provided setup script to prepare the `verl` environment for FS-GRPO:

```bash
bash setup_fs-grpo_env.sh
```

The script:

- creates a Python 3.10 conda environment
- installs `vllm`, `flash-attn`, and the training dependencies
- pins `transformers` to the 4.x series
- installs `verl` in editable mode
- optionally logs into Weights & Biases and Hugging Face if `WANDB_API_KEY` or `HF_TOKEN` are set

After setup, activate the environment with:

```bash
conda activate verl
```

## FS-GRPO Training

FS-GRPO is run with `verl` and starts from an SFT checkpoint.

Example entrypoint:

```bash
bash train-fs-grpo.sh --sft_model <path-to-sft-checkpoint> --data_dir <fs-grpo-data-dir> --output_dir <save-dir>
```

The checked-in FS-GRPO setup uses:

- `algorithm.adv_estimator=grpo`
- 8 sampled responses per prompt
- adaptive reward from [adaptive_reward.py](reward_functions/adaptive_reward.py)
- correctness reward with format shaping
- response-length regularization
- diversity bonus with cosine decay over training

The reward logic is designed to reflect the paper's FS-GRPO objective: the model should preserve correctness while learning when direct answer, perception-only output, or full reasoning is most appropriate.

Model checkpoints and parquet data are expected to be prepared separately before running FS-GRPO.

## Typical Workflow

1. Prepare the SFT data with AVR response-format annotations.
2. Run SFT to obtain a structured-response checkpoint.
3. Prepare FS-GRPO parquet data with prompts, images, and ground-truth answers.
4. Launch FS-GRPO from the SFT checkpoint.
5. Merge the final actor checkpoint to Hugging Face format if needed.

## Notes

- `train-sft.sh` and several helper scripts contain environment-specific paths that may need to be adjusted before release.
- The repository includes both LLaMA-Factory and `verl` as submodules for reproducibility.
- The checked-in FS-GRPO reward implementation is the single maintained reward path for this repository.
