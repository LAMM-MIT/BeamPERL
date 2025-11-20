#!/usr/bin/env python3
"""
Standalone evaluation script that replicates the evaluation done in DatasetEvaluationCallback.

This script evaluates a model on a dataset and computes:
- Accuracy@1: accuracy of the first generation
- Majority@k: accuracy when majority of generations are correct
- Average accuracy: average across all generations
- Format score: average format reward across all generations
"""

import argparse
import logging
import numpy as np
import os
import sys
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Add parent directory to path to import beamrl modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from beamrl.utils import SYSTEM_PROMPT, RL_POST_TRAIN_CONFIG_MAP
from beamrl.rewards import accuracy_reward, format_reward

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path, use_peft=False, adapter_path=None, max_model_len=2048, gpu_memory_utilization=0.7, dtype="bfloat16"):
    """Load model using vLLM and tokenizer from checkpoint."""
    logger.info(f"Loading model from: {model_path} using vLLM")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Handle pad token based on model type
    if "Llama" in model_path:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    elif "Qwen" in model_path:
        tokenizer.pad_token = "<|fim_pad|>"
    
    # Note: vLLM requires merged models (not PEFT adapters)
    # If use_peft is True, the adapter should already be merged before calling this function
    if use_peft and adapter_path:
        logger.warning("vLLM requires merged models. Ensure adapter is merged before loading.")
        logger.info(f"Expected merged model path: {model_path}")
    
    # Load model with vLLM
    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        trust_remote_code=True
    )
    
    logger.info("Model loaded successfully with vLLM")
    
    return llm, tokenizer


def load_eval_dataset(eval_dataset_name, eval_dataset_config=None, eval_split="test", max_eval_samples=None):
    """Load and prepare evaluation dataset."""
    # Resolve dataset name from config map if needed
    dataset_name = RL_POST_TRAIN_CONFIG_MAP.get(eval_dataset_name, eval_dataset_name)
    
    logger.info(f"Loading evaluation dataset: {dataset_name} (split: {eval_split})")
    
    # Load dataset
    if eval_dataset_config is not None:
        eval_dataset = load_dataset(
            dataset_name,
            split=eval_split,
            name=eval_dataset_config
        )
    else:
        eval_dataset = load_dataset(dataset_name, split=eval_split)
    
    # Handle column name variations
    if 'solution' not in eval_dataset.column_names and 'answer' in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_column('answer', 'solution')
    if 'problem' not in eval_dataset.column_names and 'question' in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_column('question', 'problem')
    if 'problem' not in eval_dataset.column_names and 'prompt' in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_column('prompt', 'problem')
    if 'problem' not in eval_dataset.column_names and 'query' in eval_dataset.column_names:
        eval_dataset = eval_dataset.rename_column('query', 'problem')
    
    # Limit dataset size if specified
    if max_eval_samples is not None and len(eval_dataset) > max_eval_samples:
        logger.info(f"Limiting evaluation to {max_eval_samples} samples")
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    
    logger.info(f"Loaded {len(eval_dataset)} evaluation samples")
    return eval_dataset


def evaluate_dataset(
    llm,
    tokenizer,
    eval_dataset,
    system_prompt=SYSTEM_PROMPT,
    max_prompt_length=512,
    max_generation_length=4096,
    batch_size=8,
    num_generations=5,
    temperature=0.6
):
    """Evaluate the model on the full dataset using vLLM."""
    # Store all generations per question for majority@k evaluation
    all_question_accuracy_rewards = []  # List of lists: [question_idx][generation_idx]
    all_format_rewards = []
    # Store data for logging
    all_prompts = []
    all_solutions = []
    all_generations = []  # List of lists: [question_idx][generation_idx]
    
    # Create sampling params for vLLM
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_generation_length,
        n=num_generations,
        stop=None
    )
    
    # Process dataset in batches
    for i in tqdm(range(0, len(eval_dataset), batch_size), desc="Evaluating"):
        batch_dict = eval_dataset[i:i + batch_size]
        
        # Turn dict-of-lists into list-of-dicts
        batch = [
            {key: batch_dict[key][idx] for key in batch_dict.keys()}
            for idx in range(len(batch_dict[list(batch_dict.keys())[0]]))
        ]
        
        batch_prompts = []
        batch_solutions = []
        # --- Build prompts and solutions ---
        for example in batch:
            # problem: string or list
            if isinstance(example["problem"], str):
                problem_text = example["problem"]
            elif isinstance(example["problem"], list) and len(example["problem"]) > 0:
                problem_text = example["problem"][0]
            else:
                problem_text = str(example["problem"])
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{problem_text}<think>"}
            ]
            input_text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
            batch_prompts.append(input_text)
            
            # solution / answer
            if "solution" in example:
                solution = example["solution"]
            elif "answer" in example:
                solution = example["answer"]
            else:
                solution = ""
            
            if isinstance(solution, list):
                batch_solutions.append(solution)
            else:
                batch_solutions.append([solution] if solution else [])
        
        # Generate with vLLM
        # vLLM handles batching internally and returns one RequestOutput per input prompt
        # Each RequestOutput has .outputs which is a list of CompletionOutput objects (length = n)
        outputs = llm.generate(batch_prompts, sampling_params)
        
        # Process vLLM outputs
        # vLLM returns a list of RequestOutput objects, one per input prompt
        # Each RequestOutput has .outputs which is a list of CompletionOutput objects
        # Each CompletionOutput has .text with the generated text
        batch_generations_text = []
        
        for output in outputs:
            # Extract all completions for this prompt
            prompt_generations = []
            for completion_output in output.outputs:
                prompt_generations.append(completion_output.text)
            batch_generations_text.append(prompt_generations)
        
        # Build question_completions format for reward functions
        question_completions = []
        for q_idx in range(len(batch_prompts)):
            q_completions = []
            for gen_text in batch_generations_text[q_idx]:
                q_completions.append([{"content": gen_text}])
            question_completions.append(q_completions)
        
        # --- Rewards per question ---
        batch_question_accuracy_rewards = []
        batch_format_rewards = []
        
        for question_completions_list, solution in zip(question_completions, batch_solutions):
            # Format rewards for all generations of this question
            format_rewards = format_reward(question_completions_list)
            batch_format_rewards.append(format_rewards)
            
            # Accuracy rewards
            if solution and len(solution) > 0:
                if isinstance(solution[0], str):
                    solution_terms = solution
                else:
                    solution_terms = [str(s) for s in solution]
                
                solutions_repeated = [solution_terms] * num_generations
                
                question_accuracy_rewards = accuracy_reward(
                    question_completions_list,
                    solutions_repeated,
                )
            else:
                question_accuracy_rewards = [0.0] * num_generations
            
            batch_question_accuracy_rewards.append(question_accuracy_rewards)
        
        # Aggregate
        all_question_accuracy_rewards.extend(batch_question_accuracy_rewards)
        all_format_rewards.extend(batch_format_rewards)
        all_prompts.extend(batch_prompts)
        all_solutions.extend(batch_solutions)
        all_generations.extend(batch_generations_text)
    
    # Compute aggregate metrics
    # Accuracy@1: accuracy of first generation
    accuracy_at_1 = np.mean([rewards[0] for rewards in all_question_accuracy_rewards]) if all_question_accuracy_rewards else 0.0
    
    # Majority@k: if majority (>= ceil(k/2)) of generations are correct, count as correct
    majority_threshold = (num_generations + 1) // 2  # e.g., 3 out of 5
    majority_correct = []
    for question_rewards in all_question_accuracy_rewards:
        num_correct = sum(question_rewards)
        majority_correct.append(1.0 if num_correct >= majority_threshold else 0.0)
    accuracy_majority = np.mean(majority_correct) if majority_correct else 0.0
    
    # Average accuracy across all generations
    all_individual_rewards = [r for rewards in all_question_accuracy_rewards for r in rewards]
    accuracy_avg = np.mean(all_individual_rewards) if all_individual_rewards else 0.0
    
    # Format score: average across all generations
    all_format_scores = [np.mean(f) for f in all_format_rewards]
    format_score = np.mean(all_format_scores) if all_format_scores else 0.0
    
    return {
        "accuracy_at_1": float(accuracy_at_1),
        "accuracy_majority": float(accuracy_majority),
        "accuracy_avg": float(accuracy_avg),
        "format_score": float(format_score),
        "num_samples": len(all_question_accuracy_rewards),
        "num_generations_per_sample": num_generations,
        # Data for detailed logging
        "prompts": all_prompts,
        "solutions": all_solutions,
        "generations": all_generations,
        "accuracy_rewards": all_question_accuracy_rewards,
        "format_rewards": all_format_rewards
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on a dataset")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model checkpoint (must be merged model for vLLM)")
    parser.add_argument("--use_peft", action="store_true",
                        help="Whether the model uses PEFT adapters (adapter must be merged before loading)")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to PEFT adapter (note: vLLM requires merged models)")
    parser.add_argument("--max_model_len", type=int, default=2048,
                        help="Maximum model length for vLLM (default: 2048)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                        help="GPU memory utilization for vLLM (default: 0.7)")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Data type for vLLM (default: bfloat16)")
    
    # Dataset arguments
    parser.add_argument("--eval_dataset_name", type=str, required=True,
                        help="Name of the evaluation dataset (from RL_POST_TRAIN_CONFIG_MAP or HuggingFace Hub)")
    parser.add_argument("--eval_dataset_config", type=str, default=None,
                        help="Optional dataset configuration name")
    parser.add_argument("--eval_split", type=str, default="test",
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (None = all)")
    
    # Generation arguments
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum prompt length (default: 512)")
    parser.add_argument("--max_generation_length", type=int, default=4096,
                        help="Maximum generation length (default: 4096)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation (default: 8)")
    parser.add_argument("--num_generations", type=int, default=5,
                        help="Number of generations per question (default: 5)")
    parser.add_argument("--temperature", type=float, default=0.6,
                        help="Temperature for generation (default: 0.6)")
    
    # Output arguments
    parser.add_argument("--output_file", type=str, default=None,
                        help="Optional file to save detailed results (JSON format)")
    parser.add_argument("--log_wandb", action="store_true",
                        help="Log results to WandB")
    parser.add_argument("--wandb_project", type=str, default="beamrl-eval",
                        help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="WandB run name")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.use_peft and args.adapter_path is None:
        raise ValueError("--adapter_path is required when --use_peft is True")
    
    # Initialize WandB if requested
    if args.log_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"eval_{args.eval_dataset_name}",
            config=vars(args)
        )
    
    # Load model and tokenizer using vLLM
    llm, tokenizer = load_model_and_tokenizer(
        args.model_path,
        use_peft=args.use_peft,
        adapter_path=args.adapter_path,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype
    )
    
    # Load evaluation dataset
    eval_dataset = load_eval_dataset(
        args.eval_dataset_name,
        eval_dataset_config=args.eval_dataset_config,
        eval_split=args.eval_split,
        max_eval_samples=args.max_eval_samples
    )
    
    # Run evaluation
    logger.info("Starting evaluation...")
    metrics = evaluate_dataset(
        llm=llm,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        system_prompt=SYSTEM_PROMPT,
        max_prompt_length=args.max_prompt_length,
        max_generation_length=args.max_generation_length,
        batch_size=args.batch_size,
        num_generations=args.num_generations,
        temperature=args.temperature
    )
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {args.eval_dataset_name} ({args.eval_split})")
    print(f"Model: {args.model_path}")
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"Generations per sample: {metrics['num_generations_per_sample']}")
    print(f"\nMetrics:")
    print(f"  Accuracy@1: {metrics['accuracy_at_1']:.4f}")
    print(f"  Majority@{args.num_generations}: {metrics['accuracy_majority']:.4f}")
    print(f"  Average Accuracy: {metrics['accuracy_avg']:.4f}")
    print(f"  Format Score: {metrics['format_score']:.4f}")
    print("="*60 + "\n")
    
    # Log to WandB if requested
    if args.log_wandb:
        wandb_metrics = {
            f"eval/accuracy_at_1": metrics["accuracy_at_1"],
            f"eval/accuracy_majority@{args.num_generations}": metrics["accuracy_majority"],
            f"eval/accuracy_avg": metrics["accuracy_avg"],
            f"eval/format_score": metrics["format_score"],
            f"eval/num_samples": metrics["num_samples"],
            f"eval/num_generations_per_sample": metrics["num_generations_per_sample"],
        }
        wandb.log(wandb_metrics)
        
        # Create WandB table with detailed results
        columns = ["sample_idx", "prompt", "solution"]
        for gen_idx in range(args.num_generations):
            columns.extend([
                f"generation_{gen_idx + 1}",
                f"accuracy_gen_{gen_idx + 1}",
                f"format_gen_{gen_idx + 1}",
            ])
        eval_table = wandb.Table(columns=columns)
        
        for idx, (prompt, solution, generations, acc_rewards, fmt_rewards) in enumerate(
            zip(
                metrics["prompts"],
                metrics["solutions"],
                metrics["generations"],
                metrics["accuracy_rewards"],
                metrics["format_rewards"],
            )
        ):
            row_values = [
                idx,
                prompt,
                str(solution) if solution else "",
            ]
            
            for gen_idx in range(args.num_generations):
                gen_text = generations[gen_idx] if gen_idx < len(generations) else ""
                acc = acc_rewards[gen_idx] if gen_idx < len(acc_rewards) else 0.0
                fmt = fmt_rewards[gen_idx] if gen_idx < len(fmt_rewards) else 0.0
                row_values.extend([gen_text, acc, fmt])
            
            eval_table.add_data(*row_values)
        
        wandb.log({"eval/generations_table": eval_table})
        wandb.finish()
    
    # Save detailed results to file if requested
    if args.output_file:
        import json
        # Convert numpy types to native Python types for JSON serialization
        output_metrics = {
            "accuracy_at_1": metrics["accuracy_at_1"],
            "accuracy_majority": metrics["accuracy_majority"],
            "accuracy_avg": metrics["accuracy_avg"],
            "format_score": metrics["format_score"],
            "num_samples": metrics["num_samples"],
            "num_generations_per_sample": metrics["num_generations_per_sample"],
            "prompts": metrics["prompts"],
            "solutions": [str(s) if isinstance(s, list) else s for s in metrics["solutions"]],
            "generations": metrics["generations"],
            "accuracy_rewards": metrics["accuracy_rewards"],
            "format_rewards": metrics["format_rewards"],
        }
        with open(args.output_file, 'w') as f:
            json.dump(output_metrics, f, indent=2)
        logger.info(f"Detailed results saved to {args.output_file}")


if __name__ == "__main__":
    main()