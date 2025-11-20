import copy
import logging
import numpy as np
import pandas as pd
import shutil
import torch
from transformers import TrainerCallback
import wandb
from concurrent.futures import Future
from huggingface_hub import create_branch, create_repo, list_repo_commits, upload_folder
from datasets import load_dataset
from tqdm import tqdm

from beamrl.utils import SYSTEM_PROMPT, RL_POST_TRAIN_CONFIG_MAP
from beamrl.rewards import accuracy_reward, format_reward

logger = logging.getLogger(__name__)


class DummyConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class DatasetEvaluationCallback(TrainerCallback):
    """
    Evaluates the model on a full dataset from HuggingFace Hub during training.
    Computes accuracy and format metrics and logs them to WandB.
    """
    def __init__(self,
                 eval_dataset_name: str,
                 eval_dataset_config: str | None = None,
                 eval_split: str = "test",
                 system_prompt=SYSTEM_PROMPT,
                 max_prompt_length=512,
                 max_generation_length=4096,
                 eval_steps=500,
                 batch_size=8,
                 max_eval_samples: int | None = None,
                 symbol_regex: str = "P",
                 num_generations: int = 5,
                 temperature: float = 0.6):
        """
        Args:
            eval_dataset_name: Name of the dataset from HuggingFace Hub (or key from RL_POST_TRAIN_CONFIG_MAP)
            eval_dataset_config: Optional dataset configuration name
            eval_split: Split to evaluate on (default: "test")
            system_prompt: System prompt to use for evaluation
            max_generation_length: Maximum generation length
            eval_steps: How often to run evaluation (in training steps)
            batch_size: Batch size for evaluation
            max_eval_samples: Maximum number of samples to evaluate (None = all)
            symbol_regex: Symbol regex pattern for accuracy evaluation (default: "P")
            num_generations: Number of generations per question for majority@k evaluation (default: 5)
            temperature: Temperature for generation (default: 0.6)
        """
        self.eval_dataset_name = eval_dataset_name
        self.eval_dataset_config = eval_dataset_config
        self.eval_split = eval_split
        self.system_prompt = system_prompt
        self.max_prompt_length = max_prompt_length
        self.max_generation_length = max_generation_length
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self.max_eval_samples = max_eval_samples
        self.symbol_regex = symbol_regex
        self.num_generations = num_generations
        self.temperature = temperature
        
        self.eval_dataset = None
        self.tokenizer = None

        # For a single, growing WandB table
        self.eval_table = None
        self.eval_table_next_idx = 0  # global row/sample index across evals

    def on_init_end(self, args, state, control, processing_class=None, **kwargs):
        """Load the evaluation dataset and prepare tokenizer."""
        if state.is_world_process_zero:
            tokenizer = processing_class
            self.tokenizer = tokenizer
            
            # Resolve dataset name from config map if needed
            dataset_name = RL_POST_TRAIN_CONFIG_MAP.get(self.eval_dataset_name, self.eval_dataset_name)
            
            logger.info(f"Loading evaluation dataset: {dataset_name} (split: {self.eval_split})")
            
            # Load dataset
            if self.eval_dataset_config is not None:
                self.eval_dataset = load_dataset(
                    dataset_name, 
                    split=self.eval_split, 
                    name=self.eval_dataset_config
                )
            else:
                self.eval_dataset = load_dataset(dataset_name, split=self.eval_split)
            
            # Handle column name variations
            if 'solution' not in self.eval_dataset.column_names and 'answer' in self.eval_dataset.column_names:
                self.eval_dataset = self.eval_dataset.rename_column('answer', 'solution')
            if 'problem' not in self.eval_dataset.column_names and 'question' in self.eval_dataset.column_names:
                self.eval_dataset = self.eval_dataset.rename_column('question', 'problem')
            if 'problem' not in self.eval_dataset.column_names and 'prompt' in self.eval_dataset.column_names:
                self.eval_dataset = self.eval_dataset.rename_column('prompt', 'problem')
            if 'problem' not in self.eval_dataset.column_names and 'query' in self.eval_dataset.column_names:
                self.eval_dataset = self.eval_dataset.rename_column('query', 'problem')
            
            # Limit dataset size if specified
            if self.max_eval_samples is not None and len(self.eval_dataset) > self.max_eval_samples:
                logger.info(f"Limiting evaluation to {self.max_eval_samples} samples")
                self.eval_dataset = self.eval_dataset.select(range(self.max_eval_samples))
            
            logger.info(f"Loaded {len(self.eval_dataset)} evaluation samples")

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        """Run evaluation on the dataset periodically."""
        if state.global_step % self.eval_steps == 0:
            if state.is_world_process_zero and self.eval_dataset is not None:
                logger.info(f"Running dataset evaluation at step {state.global_step}")
                metrics = self.evaluate_dataset(model, processing_class)
                
                # Log metrics to WandB
                wandb_metrics = {
                    f"eval/accuracy_at_1": metrics["accuracy_at_1"],
                    f"eval/accuracy_majority@{self.num_generations}": metrics["accuracy_majority"],
                    f"eval/accuracy_avg": metrics["accuracy_avg"],
                    f"eval/format_score": metrics["format_score"],
                    f"eval/num_samples": metrics["num_samples"],
                    f"eval/num_generations_per_sample": metrics["num_generations_per_sample"],
                    f"eval/step": state.global_step,
                }
                wandb.log(wandb_metrics)
                
                logger.info(f"Evaluation metrics at step {state.global_step}: {wandb_metrics}")
                
                # Create the WandB table once, then append rows each eval
                if self.eval_table is None:
                    columns = ["step", "sample_idx", "prompt", "solution"]
                    for gen_idx in range(self.num_generations):
                        columns.extend([
                            f"generation_{gen_idx + 1}",
                            f"accuracy_gen_{gen_idx + 1}",
                            f"format_gen_{gen_idx + 1}",
                        ])
                    self.eval_table = wandb.Table(columns=columns)

                # Append this eval's rows to the existing table
                for local_idx, (prompt, solution, generations, acc_rewards, fmt_rewards) in enumerate(
                    zip(
                        metrics["prompts"],
                        metrics["solutions"],
                        metrics["generations"],
                        metrics["accuracy_rewards"],
                        metrics["format_rewards"],
                    )
                ):
                    global_idx = self.eval_table_next_idx

                    # Base columns
                    row_values = [
                        state.global_step,                 # step
                        global_idx,                        # global sample idx across evals
                        prompt,
                        str(solution) if solution else "",
                    ]

                    # Per-generation columns
                    for gen_idx in range(self.num_generations):
                        gen_text = generations[gen_idx] if gen_idx < len(generations) else ""
                        acc = acc_rewards[gen_idx] if gen_idx < len(acc_rewards) else 0.0
                        fmt = fmt_rewards[gen_idx] if gen_idx < len(fmt_rewards) else 0.0
                        row_values.extend([gen_text, acc, fmt])

                    self.eval_table.add_data(*row_values)
                    self.eval_table_next_idx += 1

                # Log the same table object (W&B will show the growing table)
                wandb.log({"eval/generations_table": self.eval_table})
                logger.info(
                    f"Appended {len(metrics['prompts'])} evaluation samples to WandB table "
                    f"(total rows: {self.eval_table_next_idx})"
                )

    def evaluate_dataset(self, model, tokenizer):
        """Evaluate the model on the full dataset."""
        if self.eval_dataset is None:
            return {
                "accuracy": 0.0, 
                "format_score": 0.0, 
                "num_samples": 0,
                "accuracy_at_1": 0.0,
                "accuracy_majority": 0.0,
                "accuracy_avg": 0.0,
                "num_generations_per_sample": self.num_generations,
                "prompts": [],
                "solutions": [],
                "generations": [],
                "accuracy_rewards": [],
                "format_rewards": []
            }
        
        # Set model to inference mode if using PEFT
        if hasattr(model, "peft_config"):
            model.peft_config['default'].inference_mode = True
        
        model.eval()

        # Store all generations per question for majority@k evaluation
        all_question_accuracy_rewards = []  # List of lists: [question_idx][generation_idx]
        all_format_rewards = []
        # Store data for logging table
        all_prompts = []
        all_solutions = []
        all_generations = []  # List of lists: [question_idx][generation_idx]
        
        # Process dataset in batches
        num_batches = (len(self.eval_dataset) + self.batch_size - 1) // self.batch_size
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.eval_dataset), self.batch_size), desc="Evaluating"):
                batch_dict = self.eval_dataset[i:i + self.batch_size]

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
                        {"role": "system", "content": self.system_prompt},
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

                # --- Tokenize once per batch ---
                tokenized = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_prompt_length,
                )
                tokenized = {k: v.to(model.device) for k, v in tokenized.items()}

                # We generate num_generations completions per prompt in ONE call
                prompt_length = tokenized["input_ids"].shape[1]  # number of input tokens

                outputs = model.generate(
                    **tokenized,
                    max_new_tokens=self.max_generation_length,
                    temperature=self.temperature,
                    do_sample=True,
                    num_return_sequences=self.num_generations,
                    # use_cache=True,
                )

                # outputs shape: (batch_size * num_generations, total_seq_len)
                # slice off prompt tokens
                gen_only_ids = outputs[:, prompt_length:]

                # Decode all at once
                decoded = tokenizer.batch_decode(gen_only_ids, skip_special_tokens=True)

                # Reshape: [batch_size][num_generations]
                batch_size_eff = len(batch)
                assert len(decoded) == batch_size_eff * self.num_generations

                question_completions = []       # [question_idx][gen_idx] -> [{"content": text}]
                batch_generations_text = []     # [question_idx][gen_idx] -> plain text

                for q_idx in range(batch_size_eff):
                    q_completions = []
                    q_texts = []
                    for gen_idx in range(self.num_generations):
                        flat_idx = q_idx * self.num_generations + gen_idx
                        completion_text = decoded[flat_idx]
                        q_completions.append([{"content": completion_text}])
                        q_texts.append(completion_text)

                        # if you still want logging, keep but it will slow things down
                        # print(f"[TPH] Generation {i + q_idx}-{gen_idx + 1}: {completion_text}")

                    question_completions.append(q_completions)
                    batch_generations_text.append(q_texts)

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

                        solutions_repeated = [solution_terms] * self.num_generations

                        question_accuracy_rewards = accuracy_reward(
                            question_completions_list,
                            solutions_repeated,
                        )
                    else:
                        question_accuracy_rewards = [0.0] * self.num_generations

                    batch_question_accuracy_rewards.append(question_accuracy_rewards)

                # Aggregate
                all_question_accuracy_rewards.extend(batch_question_accuracy_rewards)
                all_format_rewards.extend(batch_format_rewards)
                all_prompts.extend(batch_prompts)
                all_solutions.extend(batch_solutions)
                all_generations.extend(batch_generations_text)
        
        # Restore training mode
        if hasattr(model, "peft_config"):
            model.peft_config['default'].inference_mode = False
        model.train()
        
        # Compute aggregate metrics
        # Accuracy@1: accuracy of first generation
        accuracy_at_1 = np.mean([rewards[0] for rewards in all_question_accuracy_rewards]) if all_question_accuracy_rewards else 0.0
        
        # Majority@k: if majority (>= ceil(k/2)) of generations are correct, count as correct
        majority_threshold = (self.num_generations + 1) // 2  # e.g., 3 out of 5
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
            "num_generations_per_sample": self.num_generations,
            # Data for logging table
            "prompts": all_prompts,
            "solutions": all_solutions,
            "generations": all_generations,
            "accuracy_rewards": all_question_accuracy_rewards,
            "format_rewards": all_format_rewards
        }