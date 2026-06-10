from dataclasses import dataclass, field
from typing import Literal

@dataclass
class ModelPTConfig:
    # //*******Model post-training configs*******//
    model_post_train_type: Literal["grpo", "sft"] = field(default="grpo")
    model_post_train_dataset_name: str = field(default="beamrl_train")
    model_post_train_dataset_config: str | None = field(default=None)
    trace_free: bool = field(default=True)
    # Optional custom save bucket name for checkpoints and hub repo suffix
    save_name: str | None = field(default=None)
    rl_post_train_reward_funcs: list[str] = field(default_factory=lambda: ["accuracy", "format"])
    rl_post_train_reward_weights: list[str] = field(default_factory=lambda: [2.0, 1.0])


DEFAULT_CHAT_TEMPLATE = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"
REASON_CHAT_TEMPLATE = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜>'}}{% endif %}"

# problem/question, (solution), answer => combined text for uninstructed models
RL_POST_TRAIN_CONFIG_MAP = {
    "beamrl_train":    "tphage/BeamRL-TrainData",
    "beamrl_eval":     "tphage/BeamRL-EvalData",
    "beamrl_eval_v2":  "tphage/BeamRL-EvalData-v2",
}

SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user
with the answer. The reasoning process and answer are enclosed within <think> and </think> tags, and the answer final answer is put within \\boxed{{}}. I.e., <think> reasoning process here </think> answer here and then the final answer within \\boxed{{}}.
"""

def make_conv_for_grpo(example, system_prompt, num_questions=-1):
    """
    Create multiple conversation formats for GRPO training, one per question.
    
    Args:
        example: Dataset example with a "problem" field (can be string or list)
        system_prompt: System prompt to use
        num_questions: Number of questions to use from the problem list.
                      If None or -1, uses all questions. Default is -1.
    
    Returns:
        List of dictionaries, each with "prompt" key containing conversation messages.
        Each dictionary is a separate training example.
    """
    # Handle case where problem might be a string or a list
    if isinstance(example["problem"], str):
        problems = [example["problem"]]
    else:
        problems = example["problem"]
    
    # Determine how many questions to use
    if num_questions is None or num_questions == -1:
        questions_to_use = problems
    else:
        questions_to_use = problems[:num_questions]
    
    # Create one training example per question
    examples = []
    for question in questions_to_use:
        examples.append({
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'{question}<think>'},
            ],
            # Preserve other fields from the original example (like solution, answer, etc.)
            **{k: v for k, v in example.items() if k != "problem"}
        })
    
    return examples

# def make_conv_for_grpo(example, system_prompt):
#     return {
#         "prompt": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f'{example["problem"][0]}<\\think>'},
#         ]
#     }