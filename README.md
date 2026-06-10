# BeamPERL

BeamPERL is a reinforcement learning framework designed to develop self-taught language models capable of solving beam mechanics problems. It leverages Parameter-Efficient Fine-Tuning (PEFT) by applying tunable Low-Rank Adaptation (LoRA) layers to a small, distilled large reasoning model (LRM), while keeping the underlying LRM weights frozen. These LoRA layers are fine-tuned with Reinforcement Learning from Verifiable Rewards (RLVR) using a synthetic dataset of beam mechanics questions. The result is the PE-RLVR-FT BeamPERL model: a parameter-efficient, reinforcement-learning from verifiable rewards, fine-tuned large language model specialized in beam mechanics problem-solving.

## Features

- **GRPO Training**: Implements Group Relative Policy Optimization for RLFT
- **PEFT**: Supports Parameter-Efficient Fine-Tuning with LoRA adapters
- **Custom Reward Functions**: Includes custom accuracy and format-based reward functions
- **DeepSpeed Integration**: Supports distributed training with DeepSpeed ZeRO-2
- **vLLM Integration**: Uses vLLM for efficient inference during training
- **HuggingFace Hub Integration**: Automatic model pushing to HuggingFace Hub
- **WandB Logging**: Integrated experiment tracking with Weights & Biases
- **Comprehensive Evaluation**: Evaluation scripts for baseline and post-trained models on both beam mechanics and mathematical reasoning tasks

## Project Structure

```
BeamRL/
├── beamrl/                            
│   ├── grpo.py                        # Main GRPO training script
│   ├── rewards.py                     # Reward function implementations
│   ├── utils.py                       # Utility functions and configurations
│   ├── eval_callback.py               # Training callbacks for dataset evaluation
│   └── merge_post_trained_models.py   # Model merging utilities
├── recipes/                           
│   ├── train_model_beamrl.yaml        # Training configuration (combined reward)
│   ├── train_model_beamrl_format_only.yaml    # Ablation: format reward only
│   ├── train_model_beamrl_accuracy_only.yaml  # Ablation: accuracy reward only
│   ├── eval_baselines_beamrl.yaml     # Baseline evaluation config (BeamRL dataset)
│   ├── eval_baselines_lighteval.yaml  # Baseline evaluation config (LightEval tasks)
│   ├── eval_model_beamrl.yaml         # Post-trained model eval config (BeamRL dataset)
│   ├── eval_model_beamrl_v2.yaml      # Post-trained model eval config (expanded v2 dataset)
│   ├── eval_model_lighteval.yaml      # Post-trained model eval config (LightEval tasks)
│   └── zero2.yaml                     # DeepSpeed ZeRO-2 configuration
├── scripts/
│   ├── train/                         # Training scripts
│   │   └── post_train_model_grpo.sh
│   ├── eval/                          # Evaluation scripts
│   │   ├── eval_baselines_beamrl.sh   # Evaluate baseline models on BeamRL dataset
│   │   ├── eval_baselines_lighteval.sh# Evaluate baseline models on LightEval tasks
│   │   ├── eval_model_beamrl.sh       # Evaluate post-trained models on BeamRL dataset
│   │   ├── eval_model_lighteval.sh    # Evaluate post-trained models on LightEval tasks
│   │   ├── run_dataset_eval.py        # Standalone dataset evaluation script (seedable)
│   │   ├── aggregate_eval_results.py  # Aggregate multi-seed results (overall + per-category)
│   │   ├── run_eval_custom_tasks.py   # Custom LightEval task definitions
│   │   └── parse_eval_config.py       # YAML config parser for evaluation
│   └── experiments/                   # End-to-end experiment pipeline (steps 1-5)
│       ├── run_full_pipeline.sh       # Master orchestration script
│       ├── step1_gen_eval_data.sh     # Generate expanded (v2) evaluation dataset
│       ├── step2_eval_original.sh     # Multi-seed eval of original checkpoints
│       ├── step3_train_ablations.sh   # Train reward-ablation models
│       ├── step4_eval_ablations.sh    # Multi-seed eval of ablation models
│       └── step5_aggregate.sh         # Aggregate all results into CSVs
└── setup/                             # Environment setup
    ├── environment.yml                
    ├── set_vars.sh                    
    ├── set_env.sh                     
    └── prepare.sh                     
```

## Installation

### Prerequisites

- CUDA 11.8+ compatible GPU(s)
- Conda
- Python 3.10

### Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/tphage/BeamPERL.git
   cd BeamPERL/BeamRL
   ```

2. **Create and activate the conda environment**:
   ```bash
   conda create -n beamrl python=3.10
   conda activate beamrl
   ```

3. **Modify the environment variables if needed**
   
   Edit `setup/set_vars.sh` to configure:
   - `HOME_PREFIX`: Base directory for project files
   - `PROJECT_PREFIX`: Project directory location
   - `WANDB_API_KEY`: Your Weights & Biases API key
   - `HF_TOKEN`: Your HuggingFace API token

4. **Set up environment variables and download the base model**:
   ```bash
   bash ./setup/set_env.sh
   bash ./setup/prepare.sh
   ```

## Training

### Training Configuration

Training parameters are defined in YAML file in the `recipes/` directory.

The `save_name` field sets the output directory for model checkpoints, determines the W&B run name, and specifies the name used when pushing models to the HuggingFace Hub. The default is `beamrl_260101`.

### Run training

```bash
bash ./scripts/train/post_train_model_grpo.sh
```

### GRPO Trainer

The main training script (`grpo.py`) handles:
- Dataset loading and preprocessing
- Model initialization with PEFT
- GRPO trainer setup with custom reward functions
- Training loop with checkpointing
- Model pushing to HuggingFace Hub

### Reward Functions

The framework includes two main reward functions:

1. **Accuracy Reward** (`accuracy_reward`): Evaluates the correctness of mathematical solutions by comparing predicted coefficients with ground truth values.

2. **Format Reward** (`format_reward`): Checks if the model output follows the required format:
   - Reasoning enclosed in `<think>` tags
   - Final answer in `\boxed{...}` format

Reward weights can be configured in the training YAML file.

### Datasets

- **beamrl_train**: Custom beam mechanics QA dataset for training ([`tphage/BeamRL-TrainData`](https://huggingface.co/datasets/tphage/BeamRL-TrainData))
- **beamrl_eval**: Custom beam mechanics QA dataset for evaluation ([`tphage/BeamRL-EvalData`](https://huggingface.co/datasets/tphage/BeamRL-EvalData), 24 samples)
- **beamrl_eval_v2**: Expanded evaluation dataset (`tphage/BeamRL-EvalData-v2`, 123 samples across 6 categories: 30 in-distribution, 30 multi-load, 18 varying-supports, 15 distributed-load, 15 length-variation, 15 applied-moment). The original 24-sample evaluation set is included bit-identically as a subset.
- Datasets are automatically downloaded from HuggingFace using the `datasets` library.
- The framework can be extended to support additional datasets via the `RL_POST_TRAIN_CONFIG_MAP` in `utils.py`

### Reward Ablation Training

Two additional training recipes isolate the contribution of each reward component, using the same hyperparameters as the main configuration:

- `train_model_beamrl_format_only.yaml`: trains with the format reward only
- `train_model_beamrl_accuracy_only.yaml`: trains with the accuracy reward only

To run an ablation, point the training script at the corresponding recipe (e.g. by setting the config name used in `post_train_model_grpo.sh`).

## Evaluation

The framework includes evaluation capabilities for both baseline and post-trained models.

### Evaluation Scripts

1. **Baseline Model Evaluation**:
   - `eval_baselines_beamrl.sh`: Evaluates baseline models (e.g., DeepSeek-R1-Distill-Qwen-1.5B) on the beam mechanics evaluation dataset
   - `eval_baselines_lighteval.sh`: Evaluates baseline models on mathematical reasoning evaluation datasets (AIME24, AIME25, AMC23)

2. **Post-Trained Model Evaluation**:
   - `eval_model_beamrl.sh`: Evaluates post-trained models on the beam mechanics evaluation dataset
   - `eval_model_lighteval.sh`: Evaluates post-trained models on the mathematical reasoning evaluation datasets

### Evaluation Metrics

The evaluation scripts compute:
- **Pass@1**: Binary if the model passes on the first generation (average score across the evaluation dataset)
- **Majority@k**: Binary if the majority of k generations are correct (average score across the evaluation dataset)
- **Average Accuracy**: Average accuracy across all generations
- **Format Score**: Average format reward (checks for proper reasoning tags and boxed answers)

### Running Evaluation

Each evaluation script uses its own YAML configuration file in the `recipes/` directory. To run the different evaluations:

```bash
# Evaluate baseline models on BeamRL dataset
bash ./scripts/eval/eval_baselines_beamrl.sh

# Evaluate baseline models on LightEval tasks
bash ./scripts/eval/eval_baselines_lighteval.sh

# Evaluate post-trained models on BeamRL dataset
bash ./scripts/eval/eval_model_beamrl.sh

# Evaluate post-trained models on LightEval tasks
bash ./scripts/eval/eval_model_lighteval.sh
```

The evaluation scripts automatically handle:
- Model merging (for PEFT adapters)
- Model-specific configuration (max lengths, etc.)
- WandB logging
- Batch processing of multiple checkpoints or models

### Multi-Seed Evaluation

`eval_model_beamrl.sh` runs the evaluation once per seed (default seeds: 42, 123, 456) so that metrics can be reported as mean ± standard deviation. Adapters are merged once, then each merged checkpoint is evaluated under every seed. Override the seeds via:

```bash
EVAL_SEEDS="42 123" bash ./scripts/eval/eval_model_beamrl.sh
```

After all runs complete, aggregate the per-run JSON results into summary CSVs:

```bash
python ./scripts/eval/aggregate_eval_results.py --output_dir <output_dir> --per_category
```

This produces an overall summary (`aggregated_results.csv`) and, with `--per_category`, a per-category breakdown across the evaluation dataset's sample categories.

### Experiment Pipeline

`scripts/experiments/` contains the end-to-end pipeline used for the paper's expanded evaluation and reward-ablation experiments:

1. `step1_gen_eval_data.sh` — generate and upload the expanded (v2) evaluation dataset (GPU for LLM question generation; `--no-llm` for template questions)
2. `step2_eval_original.sh` — multi-seed evaluation of the original BeamPERL checkpoints on the v2 dataset
3. `step3_train_ablations.sh` — train the format-only and accuracy-only ablation models
4. `step4_eval_ablations.sh` — multi-seed evaluation of the ablation checkpoints
5. `step5_aggregate.sh` — aggregate all results into overall and per-category CSVs

`run_full_pipeline.sh` runs all five steps in sequence; in practice the steps are usually run independently (training takes days). Each script's header documents its prerequisites and GPU requirements.

## Dataset Generation

The `DataGen/` directory contains a Jupyter notebook (`dataGen.ipynb`) for generating synthetic beam mechanics datasets used for training. The dataset generation process involves: (1) creating beam configurations with varying symbolic parameters (lengths, loads, support positions), (2) solving beam equations symbolically using the SymBeam library to obtain reactions, moments, and deflections, (3) generating natural language questions using LLMs that ask about reaction forces at supports, and (4) extracting ground-truth answers from the solved beam equations. The notebook uploads the final processed dataset to the HuggingFace Hub, which can then be used for training by BeamRL.

Additionally, the `DataGen/` directory contains an evaluation dataset generation notebook (`dataGen_eval.ipynb`) for creating evaluation datasets used to assess model performance.

The expanded (v2) evaluation dataset is produced by `DataGen/generate_eval_v2.py`. It generates 6 sample categories (in-distribution, multi-load, varying-supports, distributed-load, length-variation, applied-moment), solves each configuration symbolically with the bundled `symbeam_v2` solver, generates natural-language question variants with an LLM (or templates via `--no-llm`), verifies force/moment balance for every sample, includes the original 24-sample evaluation set bit-identically as a subset, and uploads the result to the HuggingFace Hub. `DataGen/test_signed_answer_format.py` contains unit tests for the signed-answer formatting used by the generator.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for more information.

## Acknowledgments

This project is built upon the two open source repositories Tina and Open R1. The dataset generation uses a custom version of the SymBeam Python software, modified by the authors. Furthermore, we greatly appreciate the wider open source community for sharing knowledge and resources in this rapidly evolving area that is parameter efficient reinforcement learning fine tuning of large language models.

- **Tina: Tiny Reasoning Models via LoRA**
  > Wang, S., Asilis, J., Akgül, Ö. F., Bilgin, E. B., Liu, O., & Neiswanger, W. (2025). Tina: Tiny Reasoning Models via LoRA. [arXiv:2504.15777](https://arxiv.org/abs/2504.15777) [cs.CL]

- **Open R1**
  > Hugging Face. (2025). Open R1: A fully open reproduction of DeepSeek-R1. [GitHub](https://github.com/huggingface/open-r1)

- **SymBeam**
  > Carneiro, A. (2020). SymBeam: A pedagogical package for beam bending. [GitHub](https://github.com/amcc1996/symbeam)

## Citation

```bibtex
@misc{hage2026beamperlparameterefficientrlverifiable,
      title={BeamPERL: Parameter-Efficient RL with Verifiable Rewards Specializes Compact LLMs for Structured Beam Mechanics Reasoning}, 
      author={Tarjei Paule Hage and Markus J. Buehler},
      year={2026},
      eprint={2603.04124},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2603.04124}, 
}
```
