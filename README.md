# BeamPERL

BeamPERL is a reinforcement learning framework designed to develop self-taught language models capable of solving beam mechanics problems. It leverages Parameter-Efficient Fine-Tuning (PEFT) by applying tunable Low-Rank Adaptation (LoRA) layers to a small, distilled large reasoning model (LRM), while keeping the underlying LRM weights frozen. These LoRA layers are fine-tuned with Reinforcement Learning from Verifiable Rewards (RLVR) using a synthetic dataset of beam mechanics questions. The result is the PERLVRFT BeamPERL model: a parameter-efficient, reinforcement-learning, from verifiable rewards, fine-tuned large language model specialized in beam mechanics problem-solving.

## Features

- **GRPO Training**: Implements Group Relative Policy Optimization for RLFT
- **PEFT**: Supports Parameter-Efficient Fine-Tuning with LoRA adapters
- **Custom Reward Functions**: Includes custom accuracy and format-based reward functions
- **DeepSpeed Integration**: Supports distributed training with DeepSpeed ZeRO-2
- **vLLM Integration**: Uses vLLM for efficient inference during training
- **HuggingFace Hub Integration**: Automatic model pushing to HuggingFace Hub
- **WandB Logging**: Integrated experiment tracking with Weights & Biases

## Project Structure

```
BeamRL/
├── beamrl/                            
│   ├── grpo.py                        # Main GRPO training script
│   ├── rewards.py                     # Reward function implementations
│   ├── utils.py                       # Utility functions and configurations
│   ├── callback.py                    # Training callbacks
│   └── merge_post_trained_models.py   # Model merging utilities
├── recipes/                           
│   ├── train_model_beamrl.yaml           # Training configuration
│   └── zero2.yaml                     # DeepSpeed ZeRO-2 configuration
├── scripts/
│   ├── train/                         # Training scripts
│   │   └── post_train_model_grpo.sh
│   └── eval/                          # Evaluation scripts
│       ├── eval_baselines.sh
│       ├── eval_post_train.sh
│       └── run_eval_custom_tasks.py
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

- **beamrl**: Custom beam mechanics QA dataset
- **beamrl_eval**: Custom beam mechanics QA dataset
- Datasets are automatically downloaded from HuggingFace using the `datasets` library.
- The framework can be extended to support additional datasets via the `RL_POST_TRAIN_CONFIG_MAP` in `utils.py`

## Dataset Generation

The `DataGen/` directory contains a Jupyter notebook (`dataGen.ipynb`) for generating synthetic beam mechanics datasets used for training. The dataset generation process involves: (1) creating beam configurations with varying symbolic parameters (lengths, loads, support positions), (2) solving beam equations symbolically using the SymBeam library to obtain reactions, moments, and deflections, (3) generating natural language questions using LLMs that ask about reaction forces at supports, and (4) extracting ground-truth answers from the solved beam equations. The notebook uploads the final processed dataset to the HuggingFace Hub, which can then be used for training by BeamRL.

Additionally, the `DataGen/` directory contains an evaluation dataset generation notebook (`dataGen_eval.ipynb`) for creating evaluation datasets used to assess model performance.

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for more information.

## Acknowledgments

This project is built upon the two open source repositories Tina and Open R1. The dataset generation uses a modified version of the SymBeam Python software. Furthermore, we greatly appreciate the wider open source community for sharing knowledge and resources in this rapidly evolving area that is parameter efficient reinforcement learning fine tuning of large language models.

- **Tina: Tiny Reasoning Models via LoRA**
  > Wang, S., Asilis, J., Akgül, Ö. F., Bilgin, E. B., Liu, O., & Neiswanger, W. (2025). Tina: Tiny Reasoning Models via LoRA. [arXiv:2504.15777](https://arxiv.org/abs/2504.15777) [cs.CL]

- **Open R1**
  > Hugging Face. (2025, January). Open R1: A fully open reproduction of DeepSeek-R1. [Open R1](https://github.com/huggingface/open-r1)

- **SymBeam**
  > Carneiro, A. (2020). SymBeam: A pedagogical package for beam bending. [GitHub](https://github.com/amcc1996/symbeam)

## Citation

```bibtex
@misc{hage2025beamperl,
  title={BeamPERL: Parameter-Efficient Reinforcement Learning for Verifiable Beam Mechanics Problem-Solving},
  author={Tarjei P. Hage and Markus J. Buehler},
  year={2025},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
