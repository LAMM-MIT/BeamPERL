#!/usr/bin/env python3
"""
Parse YAML config file for evaluation and output bash variables.
Supports multiple evaluation modes: baselines_beamrl, baselines_lighteval, post_train_beamrl, post_train_lighteval.
"""

import argparse
import sys
import yaml


def parse_baselines_beamrl(config):
    """Parse config for baseline evaluation (beamrl)."""
    # Handle models list
    models = config.get('models', ['Qwen/Qwen2.5-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'])
    model_str = ' '.join(f'"{m}"' for m in models)
    print(f"MODEL_LIST=({model_str})")

    # Evaluation parameters
    print(f"MAX_PROMPT_LENGTH={config.get('max_prompt_length', 512)}")
    print(f"MAX_GENERATION_LENGTH={config.get('max_generation_length', 4096)}")
    print(f"BATCH_SIZE={config.get('batch_size', 8)}")
    print(f"NUM_GENERATIONS={config.get('num_generations', 5)}")
    print(f"TEMPERATURE={config.get('temperature', 0.6)}")

    # Dataset config
    print(f"EVAL_DATASET_NAME=\"{config.get('eval_dataset_name', 'beamrl_eval')}\"")
    eval_dataset_config = config.get('eval_dataset_config')
    if eval_dataset_config:
        print(f"EVAL_DATASET_CONFIG=\"{eval_dataset_config}\"")
    else:
        print("EVAL_DATASET_CONFIG=\"\"")
    print(f"EVAL_SPLIT=\"{config.get('eval_split', 'train')}\"")
    max_eval_samples = config.get('max_eval_samples')
    if max_eval_samples:
        print(f"MAX_EVAL_SAMPLES={max_eval_samples}")
    else:
        print("MAX_EVAL_SAMPLES=\"\"")

    # vLLM parameters
    print(f"MAX_MODEL_LEN={config.get('max_model_len', 32768)}")
    print(f"GPU_MEMORY_UTILIZATION={config.get('gpu_memory_utilization', 0.7)}")
    print(f"DTYPE=\"{config.get('dtype', 'bfloat16')}\"")

    # WandB config
    print(f"LOG_WANDB={str(config.get('log_wandb', True)).lower()}")
    print(f"WANDB_PROJECT=\"{config.get('wandb_project', 'beamrl-eval-baselines')}\"")

    # Model-specific max lengths (for compatibility)
    max_model_length = config.get('max_model_length', 32768)
    max_new_tokens = config.get('max_new_tokens', 32768)
    print(f"DEFAULT_MAX_MODEL_LENGTH={max_model_length}")
    print(f"DEFAULT_MAX_NEW_TOKENS={max_new_tokens}")


def parse_baselines_lighteval(config):
    """Parse config for baseline evaluation (lighteval)."""
    # Handle models list
    models = config.get('models', ['Qwen/Qwen2.5-1.5B', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'])
    model_str = ' '.join(f'"{m}"' for m in models)
    print(f"MODEL_LIST=({model_str})")

    # Handle tasks list
    tasks = config.get('tasks', ['aime24', 'aime25', 'amc23'])
    task_str = ' '.join(f'"{t}"' for t in tasks)
    print(f"TASKS=({task_str})")

    # Model-specific max lengths (defaults)
    print(f"DEFAULT_MAX_MODEL_LENGTH={config.get('max_model_length', 32768)}")
    print(f"DEFAULT_MAX_NEW_TOKENS={config.get('max_new_tokens', 32768)}")

    # vLLM parameters
    print(f"DTYPE=\"{config.get('dtype', 'bfloat16')}\"")
    print(f"GPU_MEMORY_UTILIZATION={config.get('gpu_memory_utilization', 0.7)}")

    # Generation parameters
    print(f"TEMPERATURE={config.get('temperature', 0.6)}")
    print(f"TOP_P={config.get('top_p', 0.95)}")

    # GPU configuration
    cuda_devices = config.get('cuda_visible_devices', '0,1')
    print(f"CUDA_VISIBLE_DEVICES=\"{cuda_devices}\"")


def parse_post_train_beamrl(config):
    """Parse config for post-training evaluation (beamrl)."""
    # Extract values and create bash variables
    print(f"MODEL_NAME=\"{config.get('model_name', 'DeepSeek-R1-Distill-Qwen-1.5B')}\"")
    print(f"PT_TYPE=\"{config.get('pt_type', 'grpo')}\"")
    print(f"PT_CONFIG_NAME=\"{config.get('pt_config_name', 'beamrl')}\"")
    print(f"PT_DATASET_NAME=\"{config.get('pt_dataset_name', 'beamrl_train')}\"")
    print(f"SAVE_NAME=\"{config.get('save_name', 'beamrl_260101')}\"")

    # Handle checkpoints list
    checkpoints = config.get('checkpoints', ['checkpoint-3', 'checkpoint-4'])
    ckpt_str = ' '.join(f'"{c}"' for c in checkpoints)
    print(f"CKPT_LIST=({ckpt_str})")

    # Evaluation parameters
    print(f"MAX_PROMPT_LENGTH={config.get('max_prompt_length', 512)}")
    print(f"MAX_GENERATION_LENGTH={config.get('max_generation_length', 4096)}")
    print(f"BATCH_SIZE={config.get('batch_size', 8)}")
    print(f"NUM_GENERATIONS={config.get('num_generations', 5)}")
    print(f"TEMPERATURE={config.get('temperature', 0.6)}")

    # Dataset config
    print(f"EVAL_DATASET_NAME=\"{config.get('eval_dataset_name', 'beamrl_eval')}\"")
    eval_dataset_config = config.get('eval_dataset_config')
    if eval_dataset_config:
        print(f"EVAL_DATASET_CONFIG=\"{eval_dataset_config}\"")
    else:
        print("EVAL_DATASET_CONFIG=\"\"")
    print(f"EVAL_SPLIT=\"{config.get('eval_split', 'train')}\"")
    max_eval_samples = config.get('max_eval_samples')
    if max_eval_samples:
        print(f"MAX_EVAL_SAMPLES={max_eval_samples}")
    else:
        print("MAX_EVAL_SAMPLES=\"\"")

    # vLLM parameters
    print(f"MAX_MODEL_LEN={config.get('max_model_len', 32768)}")
    print(f"GPU_MEMORY_UTILIZATION={config.get('gpu_memory_utilization', 0.7)}")
    print(f"DTYPE=\"{config.get('dtype', 'bfloat16')}\"")

    # WandB config
    print(f"LOG_WANDB={str(config.get('log_wandb', True)).lower()}")
    print(f"WANDB_PROJECT=\"{config.get('wandb_project', 'beamrl-eval')}\"")

    # Model-specific max lengths (for merge script compatibility)
    max_model_length = config.get('max_model_length', 32768)
    max_new_tokens = config.get('max_new_tokens', 32768)
    print(f"DEFAULT_MAX_MODEL_LENGTH={max_model_length}")
    print(f"DEFAULT_MAX_NEW_TOKENS={max_new_tokens}")


def parse_post_train_lighteval(config):
    """Parse config for post-training evaluation (lighteval)."""
    # Extract values and create bash variables
    print(f"MODEL_NAME=\"{config.get('model_name', 'DeepSeek-R1-Distill-Qwen-1.5B')}\"")
    print(f"PT_TYPE=\"{config.get('pt_type', 'grpo')}\"")
    print(f"PT_CONFIG_NAME=\"{config.get('pt_config_name', 'beamrl')}\"")
    print(f"PT_DATASET_NAME=\"{config.get('pt_dataset_name', 'beamrl_train')}\"")
    print(f"SAVE_NAME=\"{config.get('save_name', 'beamrl_260101')}\"")

    # Handle checkpoints list
    checkpoints = config.get('checkpoints', ['checkpoint-3', 'checkpoint-4'])
    ckpt_str = ' '.join(f'"{c}"' for c in checkpoints)
    print(f"CKPT_LIST=({ckpt_str})")

    # Handle tasks list
    tasks = config.get('tasks', ['aime24', 'aime25', 'amc23'])
    task_str = ' '.join(f'"{t}"' for t in tasks)
    print(f"TASKS=({task_str})")

    # Model-specific max lengths (defaults)
    print(f"DEFAULT_MAX_MODEL_LENGTH={config.get('max_model_length', 32768)}")
    print(f"DEFAULT_MAX_NEW_TOKENS={config.get('max_new_tokens', 32768)}")

    # vLLM parameters
    print(f"DTYPE=\"{config.get('dtype', 'bfloat16')}\"")
    print(f"GPU_MEMORY_UTILIZATION={config.get('gpu_memory_utilization', 0.7)}")

    # Generation parameters
    print(f"TEMPERATURE={config.get('temperature', 0.6)}")
    print(f"TOP_P={config.get('top_p', 0.95)}")

    # GPU configuration
    cuda_devices = config.get('cuda_visible_devices', '0,1')
    print(f"CUDA_VISIBLE_DEVICES=\"{cuda_devices}\"")


def parse_config(config_path, mode):
    """Parse YAML config and output bash variable assignments based on mode."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if mode == 'baselines_beamrl':
        parse_baselines_beamrl(config)
    elif mode == 'baselines_lighteval':
        parse_baselines_lighteval(config)
    elif mode == 'post_train_beamrl':
        parse_post_train_beamrl(config)
    elif mode == 'post_train_lighteval':
        parse_post_train_lighteval(config)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: baselines_beamrl, baselines_lighteval, post_train_beamrl, post_train_lighteval")


def main():
    parser = argparse.ArgumentParser(
        description="Parse YAML config for evaluation and output bash variables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  baselines_beamrl      - Baseline evaluation with beamrl
  baselines_lighteval   - Baseline evaluation with lighteval
  post_train_beamrl     - Post-training evaluation with beamrl
  post_train_lighteval  - Post-training evaluation with lighteval
        """
    )
    parser.add_argument("config_path", type=str, help="Path to YAML config file")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=['baselines_beamrl', 'baselines_lighteval', 'post_train_beamrl', 'post_train_lighteval'],
        help="Evaluation mode"
    )
    args = parser.parse_args()

    try:
        parse_config(args.config_path, args.mode)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config_path}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse YAML config: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

