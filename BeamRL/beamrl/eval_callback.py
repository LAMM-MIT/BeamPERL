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

from beamrl.utils import SYSTEM_PROMPT, RL_POST_TRAIN_CONFIG_MAP, FIXED_PROMPT_FOR_EVALUATION    # TO BE UPDATED
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
                 max_generation_length=4096,
                 eval_steps=500,
                 batch_size=8,
                 max_eval_samples: int | None = None,
                 symbol_regex: str = "P",
                 num_generations: int = 5):
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
        """
        self.eval_dataset_name = eval_dataset_name
        self.eval_dataset_config = eval_dataset_config
        self.eval_split = eval_split
        self.system_prompt = system_prompt
        self.max_generation_length = max_generation_length
        self.eval_steps = eval_steps
        self.batch_size = batch_size
        self.max_eval_samples = max_eval_samples
        self.symbol_regex = symbol_regex
        self.num_generations = num_generations
        
        self.eval_dataset = None
        self.tokenizer = None

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
                
                # Create and log table with prompts, generations, and per-generation metrics
                table_data = []
                for idx, (prompt, solution, generations, acc_rewards, fmt_rewards) in enumerate(
                    zip(
                        metrics["prompts"],
                        metrics["solutions"],
                        metrics["generations"],
                        metrics["accuracy_rewards"],
                        metrics["format_rewards"]
                    )
                ):
                    row = {
                        "step": state.global_step,
                        "sample_idx": idx,
                        "prompt": prompt,
                        "solution": str(solution) if solution else "",
                    }
                    # Add columns for each generation
                    for gen_idx in range(self.num_generations):
                        row[f"generation_{gen_idx + 1}"] = generations[gen_idx] if gen_idx < len(generations) else ""
                        row[f"accuracy_gen_{gen_idx + 1}"] = acc_rewards[gen_idx] if gen_idx < len(acc_rewards) else 0.0
                        row[f"format_gen_{gen_idx + 1}"] = fmt_rewards[gen_idx] if gen_idx < len(fmt_rewards) else 0.0
                    
                    table_data.append(row)
                
                if table_data:
                    df = pd.DataFrame(table_data)
                    wandb.log({f"eval/generations_table_step_{state.global_step}": wandb.Table(dataframe=df)})
                    logger.info(f"Logged {len(table_data)} evaluation samples to WandB table")

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
                batch = self.eval_dataset[i:i+self.batch_size]
                
                # Prepare prompts for batch
                batch_prompts = []
                batch_solutions = []
                
                for example in batch:
                    # Handle problem field (can be string or list)
                    if isinstance(example["problem"], str):
                        problem_text = example["problem"]
                    elif isinstance(example["problem"], list) and len(example["problem"]) > 0:
                        problem_text = example["problem"][0]
                    else:
                        problem_text = str(example["problem"])
                    
                    # Format prompt with chat template
                    messages = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f'{problem_text}<\\think>'}
                    ]
                    input_text = tokenizer.apply_chat_template(
                        messages, 
                        add_generation_prompt=True, 
                        tokenize=False
                    )
                    batch_prompts.append(input_text)
                    
                    # Get solution/answer
                    if "solution" in example:
                        solution = example["solution"]
                    elif "answer" in example:
                        solution = example["answer"]
                    else:
                        solution = ""
                    
                    # Handle solution format (can be list or string)
                    if isinstance(solution, list):
                        batch_solutions.append(solution)
                    else:
                        batch_solutions.append([solution] if solution else [])
                
                # Tokenize batch
                tokenized = tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_generation_length
                )
                tokenized = {k: v.to(model.device) for k, v in tokenized.items()}
                
                # Generate multiple times for each question
                all_completions_per_question = []  # List of lists: [question_idx][generation_idx]
                
                for gen_idx in range(self.num_generations):
                    # Generate
                    outputs = model.generate(
                        **tokenized,
                        max_length=self.max_generation_length,
                        temperature=0.01,
                        top_k=1,
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    
                    # Decode completions for this generation
                    completions = []
                    for j, output in enumerate(outputs):
                        # Extract only the generated part (after the prompt)
                        prompt_length = tokenized["input_ids"][j].shape[0]
                        generated_ids = output[prompt_length:]
                        completion_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                        completions.append([{"content": completion_text}])
                    
                    all_completions_per_question.append(completions)
                
                # Reorganize: group by question instead of by generation
                # all_completions_per_question[gen_idx][question_idx] -> question_completions[question_idx][gen_idx]
                question_completions = []
                batch_generations_text = []  # Store actual generation text for logging
                for q_idx in range(len(batch)):
                    question_completions.append([
                        all_completions_per_question[gen_idx][q_idx]
                        for gen_idx in range(self.num_generations)
                    ])
                    # Extract generation text for logging
                    generations_text = [
                        all_completions_per_question[gen_idx][q_idx][0]["content"]
                        for gen_idx in range(self.num_generations)
                    ]
                    batch_generations_text.append(generations_text)
                
                # Compute rewards for this batch
                batch_question_accuracy_rewards = []  # List of lists: [question_idx][generation_idx]
                batch_format_rewards = []
                
                for question_completions_list, solution in zip(question_completions, batch_solutions):
                    # Format reward for all generations of this question
                    format_rewards = format_reward(question_completions_list)
                    batch_format_rewards.append(format_rewards)
                    
                    # Accuracy rewards for each generation
                    if solution and len(solution) > 0:
                        # Handle solution format - could be list of strings like ["0.1P", "1.9P"]
                        if isinstance(solution[0], str):
                            solution_terms = solution
                        else:
                            solution_terms = [str(s) for s in solution]
                        
                        # Repeat solution for each generation (accuracy_reward expects one solution per completion)
                        solutions_repeated = [solution_terms] * self.num_generations
                        
                        # accuracy_reward expects completions and solutions as lists
                        question_accuracy_rewards = accuracy_reward(
                            question_completions_list,
                            solutions_repeated
                        )
                    else:
                        question_accuracy_rewards = [0.0] * self.num_generations
                    
                    batch_question_accuracy_rewards.append(question_accuracy_rewards)
                
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


class FixedPromptEvaluationCallback(TrainerCallback):
    def __init__(self,
                 system_prompt=SYSTEM_PROMPT,
                 prompt=FIXED_PROMPT_FOR_EVALUATION,
                 max_generation_length=4096, eval_steps=100):

        self.system_prompt = system_prompt
        self.prompt = prompt
        self.max_generation_length = max_generation_length
        self.eval_steps = eval_steps
        self.completion_table = {
            "step": [],
            "prompt": [],
            "completion": [],
        }

    def on_init_end(self, args, state, control, processing_class=None, **kwargs):
        tokenizer = processing_class
        messages = [{"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": self.prompt}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.tokenized_prompt = tokenizer(input_text, return_tensors="pt")

    def on_step_end(self, args, state, control, model=None, processing_class=None, **kwargs):
        if state.global_step % self.eval_steps == 0:
            if state.is_world_process_zero:
                completion = self.eval_prompt(model, processing_class)
                self.completion_table["step"].append(str(state.global_step))
                self.completion_table["prompt"].append(self.prompt)
                self.completion_table["completion"].append(completion)
                df = pd.DataFrame(self.completion_table)
                wandb.log({"completions": wandb.Table(dataframe=df)})

    def eval_prompt(self, model, tokenizer):
        if hasattr(model, "peft_config"):
            model.peft_config['default'].inference_mode = True

        self.tokenized_prompt.to(model.device)
        outputs = model.generate(
            **self.tokenized_prompt,
            max_length=self.max_generation_length,
            temperature=0.01,  # Very low temperature
            top_k=1,  # Only consider the most likely token
            top_p=1.0,  # Disable nucleus sampling or set to a high value
        )
        completion = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if hasattr(model, "peft_config"):
            model.peft_config['default'].inference_mode = False

        return completion

class PushToHubRevisionCallback(TrainerCallback):
    def __init__(self, dataset_name, use_peft):
        self.dataset_name = dataset_name
        self.use_peft = use_peft

        self.pending_futures = []  # Track pending push operations

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            global_step = state.global_step

            # Create merged model directory
            if self.use_peft:
                ckpt_model_dir = f"{args.output_dir}/checkpoint-{global_step}-merged"
                original_model = kwargs["model"]  # Don't pop it, keep it intact
                model_to_save = copy.deepcopy(original_model).merge_and_unload()
                model_to_save.save_pretrained(ckpt_model_dir)
            else:
                # this dir is already created by the HF Trainer, no need to manually save
                ckpt_model_dir = f"{args.output_dir}/checkpoint-{global_step}"

            tokenizer = kwargs.get("tokenizer") or kwargs.get("processing_class")
            if tokenizer is None:
                raise ValueError("Tokenizer or processing_class must be provided.")
            tokenizer.save_pretrained(ckpt_model_dir)

            dummy_config = DummyConfig(
                hub_model_id=args.hub_model_id,
                hub_model_revision=self.dataset_name,
                checkpoint=f"checkpoint-{global_step}",
                output_dir=ckpt_model_dir,
                dataset_name=self.dataset_name,
                hub_private_repo=getattr(args, 'hub_private_repo', True),
            )

            # Start the push operation
            future = push_to_hub_revision(
                dummy_config, extra_ignore_patterns=["*.pt"]
            )

            # Store the future and directory path for cleanup later
            self.pending_futures.append((future, ckpt_model_dir))

            # Check and clean up any completed pushes
            if self.use_peft:
                self._cleanup_completed_pushes()

        return control

    def _cleanup_completed_pushes(self):
        """Check pending futures and remove directories for completed pushes."""
        still_pending = []
        for future, dir_path in self.pending_futures:
            if future.done():
                if self.use_peft:
                    # The push is complete, safe to delete the directory
                    try:
                        shutil.rmtree(dir_path)
                        logger.info(f"\nCleaned up merged model directory: {dir_path}\n")
                    except Exception as e:
                        logger.error(f"\nFailed to clean up directory {dir_path}: {e}\n")
            else:
                # Push is still in progress, keep in pending list
                still_pending.append((future, dir_path))

        self.pending_futures = still_pending

    def on_train_end(self, args, state, control, **kwargs):
        """Make sure to clean up any remaining directories at the end of training."""
        if state.is_world_process_zero and self.use_peft:
            # Wait for all pending pushes to complete
            logger.info(f"\nCleaned up for lora models.")
            for future, dir_path in self.pending_futures:
                future.result()  # Wait for completion
                try:
                    shutil.rmtree(dir_path)
                    logger.info(f"\nCleaned up merged model directory: {dir_path}\n")
                except Exception as e:
                    logger.error(f"\nFailed to clean up directory {dir_path}: {e}\n")

            self.pending_futures = []

def push_to_hub_revision(training_args, extra_ignore_patterns=[]) -> Future:
    """Pushes the model to branch on a Hub repo."""

    # Get hub_private_repo setting, defaulting to True for backward compatibility
    private_repo = getattr(training_args, 'hub_private_repo', True)
    
    # Create a repo if it doesn't exist yet
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=private_repo, exist_ok=True)
    # Get initial commit to branch from
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        # checkpoint=training_args.checkpoint,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info(f"Created target repo at {repo_url}")
    logger.info(f"Pushing to the Hub revision {training_args.hub_model_revision} with checkpoint {training_args.checkpoint}")
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        # commit_message=f"Add {training_args.hub_model_revision} checkpoint {training_args.dataset_name}",
        commit_message=f"Add {training_args.checkpoint} checkpoint post-trained on {training_args.dataset_name}",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )

    logger.info(f"Pushed to {repo_url} revision {training_args.hub_model_revision} with checkpoint {training_args.checkpoint} successfully!")

    return future
