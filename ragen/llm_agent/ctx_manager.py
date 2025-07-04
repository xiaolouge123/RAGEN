"""
This is the context manager for the LLM agent.
author: Kangrui Wang, Zihan Wang
date: 2025-03-30
"""
from itertools import zip_longest
from datetime import datetime
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import re
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from transformers import AutoTokenizer
import hydra
from ragen.utils import register_resolvers
from ragen.env import REGISTERED_ENV_CONFIGS
from tensordict import TensorDict

from dataclasses import asdict
register_resolvers()

def get_special_tokens(tokenizer: AutoTokenizer):
    if "qwen" in tokenizer.name_or_path.lower():
        special_token = tokenizer.encode("<|im_start|>")[0]
        reward_token = tokenizer.encode("<|im_end|>")[0]
    elif "llama-3" in tokenizer.name_or_path.lower():
        special_token = 128006
        reward_token = 128009
    else:
        raise ValueError(f"Unsupported model: {tokenizer.name_or_path}")
    return special_token, reward_token

def get_masks_and_scores(input_ids: torch.Tensor, tokenizer: AutoTokenizer, all_scores: List[List[float]] = None, use_turn_scores: bool = False, enable_response_mask: bool = False):
    """
    input_ids: shape (bsz, seq_len)
    Get loss mask that only learns between <|im_start|>assistant and <|im_end|>. Currently only supports qwen.
    NOTE: important! This assumes that the input_ids starts with system and then user & assistant in alternative ways
    """
    special_token, reward_token = get_special_tokens(tokenizer)
    
    turn_starts = torch.where(input_ids == special_token, 1, 0)
    turn_indicators = torch.cumsum(turn_starts, dim=-1)
    if enable_response_mask:
        loss_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1) # only learns all assistant turns
    else:
        loss_mask = (turn_indicators > 1) # learns everything after system prompt
    response_mask = (turn_indicators % 2 == 1) & (turn_indicators > 1)
    
    score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
    if use_turn_scores:
        for idx, scores in enumerate(zip_longest(*all_scores, fillvalue=0)):
            scores = torch.tensor(scores, dtype=torch.float32)
            turn_indicator = idx * 2 + 3 # 0: pad. 1: system. 2+2n: user. 3+2n: assistant
            reward_position = (input_ids == reward_token) & (turn_indicators == turn_indicator)
            # Set the last token of the rows where all positions are False to True
            reward_position[~reward_position.any(dim=-1), -1] = True
            score_tensor[reward_position] = scores
        if "qwen" in tokenizer.name_or_path.lower():
            # for Qwen, there is a "\n" between special token and reward token, so we shift this to make sure reward is assigned to the last token of a turn
            score_tensor = score_tensor.roll(shifts=1, dims=-1)
    else:
        scores = [sum(i) for i in all_scores]
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
    score_tensor = score_tensor[:, 1:] # remove the first token
    loss_mask = loss_mask[:, :-1] # remove the last token
    response_mask = response_mask[:, :-1] # remove the last token

    return score_tensor, loss_mask, response_mask



class ContextManager:
    """
    Manages the context for LLM interactions with environments.
    Translates between environment outputs and LLM inputs, and vice versa.
    """

    def __init__(self, 
                 config,
                 tokenizer,
                 processor = None,
                 mode: str = "train",
                 ):
        """
        Initialize the ContextManager.
        Processor is used to process the image data.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
        self.action_sep = self.config.agent_proxy.action_sep
        self.special_token_list = ["<think>", "</think>", "<action>", "</action>", "<answer>", "</answer>", "<|im_start|>", "<|im_end|>"]

        self.es_cfg = self.config.es_manager[mode]
        self.env_nums = {
                env_tag: n_group * self.es_cfg.group_size
                for n_group, env_tag in zip(self.es_cfg.env_configs.n_groups, self.es_cfg.env_configs.tags)
        }
        print(self.env_nums)
        self._init_prefix_lookup()
    
    def _check_env_installed(self, env_type: str):
        if env_type not in REGISTERED_ENV_CONFIGS:
            raise ValueError(f"Environment {env_type} is not installed. Please install it using the scripts/setup_{env_type}.sh script.")

    def _init_prefix_lookup(self):
        prefix_lookup = {}
        prefixes = {}
        env_config_lookup = {}
        env_config = {}
        for env_tag, env_config in self.config.custom_envs.items():
            if env_tag not in self.es_cfg.env_configs.tags:
                continue

            self._check_env_installed(env_config.env_type)
            env_config_new = asdict(REGISTERED_ENV_CONFIGS[env_config.env_type]())
            for k,v in env_config.items():
                env_config_new[k] = v
            env_instruction = env_config_new.get("env_instruction", "")
            if env_config_new.get("grid_vocab", False):
                grid_vocab_str = "\nThe meaning of each symbol in the state is:\n" + ", ".join([f"{k}: {v}" for k, v in env_config_new["grid_vocab"].items()])
                env_instruction += grid_vocab_str
            if env_config_new.get("action_lookup", False):
                action_lookup_str = "\nYour available actions are:\n" + ", ".join([f"{v}" for k, v in env_config_new["action_lookup"].items()])
                action_lookup_str += f"\nYou can make up to {env_config_new['max_actions_per_traj']} actions, separated by the action separator \" " + self.action_sep + " \"\n"
                env_instruction += action_lookup_str
            if env_config_new.get("enable_world_info", False):
                env_instruction += f"\n<CURRENT_WORLD_INFO>\nCurrent date: {datetime.now().strftime('%Y-%m-%d')}\nCurrent time: {datetime.now().strftime('%H:%M:%S')}\n</CURRENT_WORLD_INFO>" # TODO 这里添加世界线信息还是有点问题，应该在每次模拟时添加相关信息。
            prefixes[env_tag] = env_instruction
            env_config_lookup[env_tag] = {'max_tokens': env_config.get("max_tokens", self.config.actor_rollout_ref.rollout.response_length)}

        tags = self.es_cfg.env_configs.tags
        n_groups = self.es_cfg.env_configs.n_groups
        group_size = self.es_cfg.group_size

        cur_group = 0
        for env_tag, n_group in zip(tags, n_groups):
            env_instruction = prefixes[env_tag]
            start_idx = cur_group * group_size
            end_idx = (cur_group + n_group) * group_size
            for i in range(start_idx, end_idx):
                prefix_lookup[i] = env_instruction
                env_config_lookup[i] = env_config_lookup[env_tag]
            cur_group += n_group
            
        self.prefix_lookup = prefix_lookup
        self.env_config_lookup = env_config_lookup

    def _parse_webbrowser_response(self, response: str) -> List:
        """
        return: thought, actions, answer
        """
        action_pattern = r'<think>(.*?)</think>\s*<action>(.*?)</action>'
        answer_pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>'
        if "<action>" in response:
            match = re.search(action_pattern, response, re.DOTALL)
            if not match:
                return "", [], ""
            else:
                thought =  match.group(1)
                action_str = match.group(2)
                answer = "" # 默认 actions 出现的时候 answer 是空的

                for special_token in self.special_token_list:
                    action_str = action_str.replace(special_token, "").strip()
                    thought = thought.replace(special_token, "").strip()
                
                actions = [action.strip() for action in action_str.split(self.action_sep) if action.strip()]
                max_actions = self.config.agent_proxy.max_actions_per_turn
                if len(actions) > max_actions:
                    actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                
                return thought, actions, answer
                    
        elif "<answer>" in response:
            match = re.search(answer_pattern, response, re.DOTALL)
            if not match:
                return "", [], ""
            else:
                return match.group(1), [], match.group(2)
        elif "<think>" in response:
            thought_pattern = r'<think>(.*?)</think>'
            match = re.search(thought_pattern, response, re.DOTALL)
            if not match:
                return "", [], ""
            else:
                return match.group(1), [], ""
        else:
            return "", [], ""

    def _parse_response(self, response: str, env_tag: str) -> List:
        pattern = r'<think>(.*?)</think>\s*<answer>(.*?)</answer>' if self.config.agent_proxy.enable_think else r'<answer>(.*?)</answer>'
        if env_tag in ["WebBrowser"]:
            llm_response = response
            actions = []
            answer = ""

            assert self.config.agent_proxy.enable_think == True, "WebBrowser env requires enable_think to be True"
            think_content, actions, answer = self._parse_webbrowser_response(response)
            if think_content == "":
                llm_response, actions, answer = response, [], ""
            else:
                if len(actions) >= 1:
                    action_content = (self.action_sep).join(actions)
                    actions = [action_content] # webbrowser env 可以消费多个 action 的  string，只要是 \n 分割的即可
                    llm_response = f"<think>{think_content}</think><action>{action_content}</action>"
                if answer != "":
                    llm_response = f"<think>{think_content}</think><answer>{answer}</answer>"
                    answer = f"<answer>{answer}</answer>"

            return llm_response, actions, answer
        
        match = re.search(pattern, response, re.DOTALL)
        if not match:
            # think_content, action_content, actions = "", "", [] # do not remove this kind of invalid string
            llm_response, actions = response, []
        else:
            if self.config.agent_proxy.enable_think:
                think_content, action_content = match.group(1), match.group(2)
            else:
                think_content, action_content = "", match.group(1)

                
            for special_token in self.special_token_list:
                action_content = action_content.replace(special_token, "").strip()
                think_content = think_content.replace(special_token, "").strip()
            
            actions = [action.strip() for action in action_content.split(self.action_sep) if action.strip()]
            max_actions = self.config.agent_proxy.max_actions_per_turn

            if len(actions) > max_actions:
                actions = actions[:max_actions] #Only the first MAX_ACTIONS actions are kept in the rollout.
                action_content = (" " + self.action_sep + " ").join(actions)

            llm_response = f"<think>{think_content}</think><answer>{action_content}</answer>" if self.config.agent_proxy.enable_think else f"<answer>{action_content}</answer>"
        return llm_response, actions, None
        
    def _normalize_score_tensor(self, score_tensor: torch.Tensor, env_outputs: List[Dict]) -> torch.Tensor:
        """
        Normalize the score tensor to be between 0 and 1.
        NOTE: only support score at the last token for now
        """
        assert self.config.agent_proxy.use_turn_scores == False, "Reward normalization is not supported for use_turn_scores == True"
        
        rn_cfg = self.config.agent_proxy.reward_normalization
        grouping, method = rn_cfg.grouping, rn_cfg.method
        if grouping == "state":
            group_tags = [env_output["group_id"] for env_output in env_outputs]
        elif grouping == "inductive":
            group_tags = [env_output["tag"] for env_output in env_outputs]
        elif grouping == "batch":
            group_tags = [1] * len(env_outputs)
        else:
            raise ValueError(f"Invalid grouping: {grouping}")


        if method == "mean_std":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x) # stable to bf16 than x.std()
        elif method == "mean":
            norm_func = lambda x: (x - x.mean(dim=-1, keepdim=True))
        elif method == "asym_clip":
            norm_func = lambda x: ((x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6) if x.std(dim=-1, keepdim=True).abs().max() > 1e-6 else torch.zeros_like(x)).clamp(min=-1, max=3)
        elif method == "identity":
            norm_func = lambda x: x
        else:
            raise ValueError(f"Invalid normalization method: {method}")

        # apply groupwise normalization
        group2index = {}
        for i, env_tag in enumerate(group_tags):
            if env_tag not in group2index:
                group2index[env_tag] = []
            group2index[env_tag].append(i)
        group2index = {k: torch.tensor(v) for k, v in group2index.items()}

        
        # apply penalty pre-normalization
        acc_scores = score_tensor[:, -1]
        normalized_acc_scores = acc_scores.clone()
        penalty = torch.tensor([env_output.get("penalty", 0) for env_output in env_outputs], dtype=torch.float32)
        normalized_acc_scores = normalized_acc_scores + penalty

        if len(group2index) < acc_scores.shape[0]: # the group size > 1
            for group, index in group2index.items():
                normalized_acc_scores[index] = norm_func(normalized_acc_scores[index])

        score_tensor[:, -1] = normalized_acc_scores

        return score_tensor
    
    def get_eval_lm_inputs(self, env_outputs: List[Dict]) -> DataProto:
        """
        输入是最终获取的 env_outputs，
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5, "goal": "Goal 1", "gt": "Ground Truth 1"}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        # 在 history 的第一个轮次还添加了 goal 和 gt 信息。在最后一轮还会有 final_answer (非空) 用于进行打分。 这里主要结果进行评价打分，会比下面的简单些。
        """
        # TODO 这个也可以塞到 env output 里面获取
        eval_prompt = """请根据如下问题，标准答案，测试答案，评估测试答案的正确程度。
请关注测试答案如下几方面：
1. 是否充分回答了问题中的各个方面？
2. 是否回答了正确的数值结果？
3. 是否遵循了问题中要求的输出格式？
输出得分 0 到 10 分。如果测试答案没有值，则输出 0 。

<question>{question}</question>

<ground_truth>{ground_truth}</ground_truth>

<predication>{predication}</predication>

请给出你的判断得分输出这样的格式： <score>[your score]</score>
"""     
        error_eval_prompt = """
请直接输出：<score>0</score>
"""
        max_eval_response_length = self.config.llm_reward_model_api.response_length
        max_eval_model_len = self.config.llm_reward_model_api.max_model_len
        if not max_eval_model_len or max_eval_model_len <= 0:
            max_eval_model_len = self.tokenizer.model_max_length
        max_prompt_len = max_eval_model_len - max_eval_response_length

        llm_input_texts = []
        messages_list = []
        for env_output in env_outputs:
            # print(f"env_output: {env_output['history'][0]}")
            assert "goal" in env_output['history'][0] and "gt" in env_output['history'][0], "首轮env output 没找到 goal 和 gt"
            goal = env_output['history'][0]['goal']
            gt = env_output['history'][0]['gt']
            

            final_answer = env_output['history'][-1]['final_answer'] if "final_answer" in env_output['history'][-1]  and env_output['history'][-1]['final_answer'] != "" else "没有找到问题答案。"

            if gt == "" or goal == "":
                print(f"首轮env output 没找到 goal 和 gt, 请检查 {env_output['history'][0]}")
                messages = [
                    {"role": "system", "content": f"You're a helpful assistant. "}, 
                    {"role": "user", "content": error_eval_prompt}
                ]
            else:
                messages = [
                    {"role": "system", "content": f"You're a helpful assistant. "}, 
                    {"role": "user", "content": eval_prompt.format(question=goal, ground_truth=gt, predication=final_answer)}
                ]

            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            # 暂时不考虑 prompt 过长问题
            llm_input_texts.append(text)
            messages_list.append(messages)
        
        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) 
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }, batch_size=input_ids.shape[0])

        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "env_tags": np.array([env_output["tag"] for env_output in env_outputs], dtype=object), # 用来控制后续的 llm response 解析
            "messages_list": np.array(messages_list, dtype=object),
            "lm_input_texts": np.array(llm_input_texts, dtype=object), # 用来debug 看日志的
        }
        return llm_inputs

    
    def get_lm_inputs(self, env_outputs: List[Dict], prepare_for_update: bool) -> DataProto:
        """
        env_outputs - please see below example
        [
            {"env_id": 1, "history": [{"state": "###\n#x_#", "llm_response": "Response 1", "reward": 0.5}, {"state": "###\n#x_#"}]},
            {"env_id": 2, "history": [{"state": "###\n#x_#"}]},
            ...
        ]
        prefix_lookup - from env_id to initial prompt
        """
        max_response_length = self.config.actor_rollout_ref.rollout.response_length
        max_model_len = self.config.actor_rollout_ref.rollout.max_model_len
        # a safe guard for max_model_len
        if not max_model_len or max_model_len <= 0:
            max_model_len = self.tokenizer.model_max_length
        max_prompt_len = max_model_len - max_response_length

        llm_input_texts = []
        messages_list = [] # for api calling
        for env_output in env_outputs:
            env_id = env_output["env_id"]
            if 'state' in env_output['history'][-1] and prepare_for_update:
                env_output['history'] = env_output['history'][:-1] # when prepare for update, we do not add the state from the n+1 turn to the trajectory
            messages = [
                {"role": "system", "content": f"You're a helpful assistant. "}, 
                {"role": "user", "content": self.prefix_lookup[env_output["env_id"]]}
            ]
            env_tag = env_output["tag"]

            for idx, content in enumerate(env_output["history"]):
                if env_tag in ["WebBrowser"]:
                    if idx == 0 and content.get("goal", None):
                        messages[-1]["content"] += f"\n{content['goal']}" # 只在首轮次 user prompt 最后添加任务目标。
                    if idx == 0:
                        LENGTH_PROMPT = f"\nMax response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                        messages[-1]["content"] += LENGTH_PROMPT # 只在首轮次 user prompt 最后添加回复长度限制。format prompt 已经在配置文件中添加了。

                    messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                    if "state" in content:
                        if idx + 1 < len(env_output["history"]):
                            # 说明这是最后一个轮次之前的 state， 只需要添加 condensed observation
                            messages[-1]["content"] += f"State:\n{content['condensed_state']}"
                        else:
                            messages[-1]["content"] += f"State:\n{content['state']}"
                        
                    if "llm_response" in content:
                        messages.append({"role": "assistant", "content": content["llm_response"]})
                    
                    if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1): # NOTE 在 context 显示的添加 reward 应该也不是必须的吧
                        # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                        messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
                
                else:
                    messages[-1]["content"] += f"\nTurn {idx + 1}:\n"
                    # TODO 这里很奇怪额。如果 history 是多轮次的的，FORMAR_PROMPT 和 LENGTH_PROMPT 会在每个 turn 都重复，这明显没太大的必要。
                    if "state" in content:
                        FORMAT_PROMPT = "<think> [Your thoughts] </think> <answer> [your answer] </answer>" if self.config.agent_proxy.enable_think else "<answer> [your answer] </answer>"
                        LENGTH_PROMPT = f"Max response length: {self.env_config_lookup[env_output['env_id']]['max_tokens']} words (tokens)."
                        messages[-1]["content"] += f"State:\n{content['state']}\nYou have {content['actions_left']} actions left. Always output: {FORMAT_PROMPT} with no extra text. Strictly follow this format. {LENGTH_PROMPT}\n"
                    if "llm_response" in content:
                        messages.append({"role": "assistant", "content": content["llm_response"]})
                    if "reward" in content and not (prepare_for_update and idx == len(env_output["history"]) - 1):
                        # when prepare for update, we do not add the reward from the n+1 turn to the trajectory
                        messages.append({"role": "user", "content": f"Reward:\n{content['reward']}\n"})
                    

            # NOTE: this assertion is important for loss mask computation        
            assert all(msg["role"] == "assistant" for msg in messages[2::2])
            
            # NOTE: hard code 强制 prompt 不要过程，截断只截断最后一轮次user 的 state 内容
            # Truncate the prompt from the last user turn's state if it exceeds max_prompt_len
            temp_input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=True)
            if len(temp_input_ids) > max_prompt_len:
                print(f'[DEBUG] truncate prompt, temp_input_ids length: {len(temp_input_ids)}, max_prompt_len: {max_prompt_len}')
                last_user_idx = -1
                # Find the last message from a user，the last turn suppose to be the user turn
                for i in range(len(messages) - 1, -1, -1):
                    if messages[i]['role'] == 'user':
                        last_user_idx = i
                        break
                
                if last_user_idx != -1:
                    original_content = messages[last_user_idx]['content']
                    # The state information is expected to be at the end, after "State:\n".
                    # We will truncate the content that follows this marker.
                    split_marker = "State:\n"
                    parts = original_content.rsplit(split_marker, 1)

                    if len(parts) == 2:
                        base_content, state_content = parts
                        base_content += split_marker  # Restore the marker to the base part

                        # Create a copy of messages to calculate the base prompt length without the state
                        temp_messages = list(messages)
                        temp_messages[last_user_idx] = {'role': 'user', 'content': base_content}

                        # Calculate the length of the prompt without the state to determine remaining space
                        base_prompt_ids = self.tokenizer.apply_chat_template(temp_messages, add_generation_prompt=(not prepare_for_update), tokenize=True)
                        remaining_len = max_prompt_len - len(base_prompt_ids)
                        print(f'[DEBUG] env_id: {env_id} remaining_len: {remaining_len}, base_prompt_ids length: {len(base_prompt_ids)}')
                        if remaining_len > 0:
                            # Tokenize the state and truncate it from the beginning to keep the most recent info
                            cnt = 5
                            while cnt > 0:
                                state_ids = self.tokenizer.encode(state_content)
                                truncated_state_ids = state_ids[-remaining_len:]
                                print(f'[DEBUG] env_id: {env_id} truncated_state_ids length: {len(truncated_state_ids)} in cnt: {cnt}')
                                truncated_state_content = self.tokenizer.decode(truncated_state_ids, skip_special_tokens=True)
                                truncated_state_content_ids = self.tokenizer.encode(truncated_state_content)
                                if len(truncated_state_content_ids) <= remaining_len:
                                    break
                                state_content = truncated_state_content
                                cnt -= 1
                            # Reconstruct the final content for the last user message
                            messages[last_user_idx]['content'] = base_content + truncated_state_content
                        else:
                            # If there's no space for the state, truncate it completely
                            # TODO 这里也很有问题啊，如果前面内容太长，这里也很容易超长。 64K 训练很必要，或者截断前面的历史
                            messages[last_user_idx]['content'] = base_content
                        tmp_input_ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=True)
                        print(f'[DEBUG] env_id: {env_id} before truncate length: {len(temp_input_ids)} after truncate length: {len(tmp_input_ids)}')
                else:
                    print(f'[DEBUG] last_user_idx: {last_user_idx}, last turn in messages is not user turn')

            text = self.tokenizer.apply_chat_template(messages, add_generation_prompt=(not prepare_for_update), tokenize=False)
            print(f'[DEBUG] env_id: {env_id} tokenized temp_input_ids length : {len(temp_input_ids)} text length after: {len(text)}')
            # print(f'[DEBUG] messages: {messages}')
            # print(f'[DEBUG] text: {text}')
            if not prepare_for_update: # 这里的逻辑不影响 async rollout
                if self.config.agent_proxy.enable_think: # TODO 这里代码和配置文件耦合了，envs.yaml 配置文件中，对于输出的限定可能花样更多。 处理不好就有可能出错。
                    text += "<think>" # force the LLM to think before answering
                else:
                    text += "<answer>" # force the LLM to answer
            llm_input_texts.append(text)
            messages_list.append(messages) # NOTE: messages_list 这里如何实现类似在生成时 add_generation_prompt 的约束。

        inputs = self.tokenizer(llm_input_texts, return_tensors="pt", padding=True, padding_side="left", truncation=False) # do not truncate here. Process later at TODO
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
        position_ids = attention_mask.cumsum(dim=-1)
        if prepare_for_update:
            scores = [[i['reward'] for i in env_output['history']] for env_output in env_outputs]
            score_tensor, loss_mask, response_mask = get_masks_and_scores(input_ids, self.tokenizer, scores, use_turn_scores=self.config.agent_proxy.use_turn_scores, enable_response_mask=self.config.enable_response_mask) # 这种 Step 级别的 reward 后面再想想到底怎么搞吧。

            normalized_score_tensor = score_tensor
            if not self.config.agent_proxy.use_turn_scores:
                normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
            response_length = response_mask.sum(dim=-1).float().mean().item()

        llm_inputs = DataProto()
        llm_inputs.batch = TensorDict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": input_ids[:, 1:], # remove the first token
        }, batch_size=input_ids.shape[0])

        if prepare_for_update:
            llm_inputs.batch["loss_mask"] = loss_mask # remove the first token
            llm_inputs.batch["rm_scores"] = normalized_score_tensor # remove the first token
            llm_inputs.batch["original_rm_scores"] = score_tensor # remove the first token
        llm_inputs.non_tensor_batch = {
            "env_ids": np.array([env_output["env_id"] for env_output in env_outputs], dtype=object),
            "group_ids": np.array([env_output["group_id"] for env_output in env_outputs], dtype=object),
            "env_tags": np.array([env_output["tag"] for env_output in env_outputs], dtype=object), # 用来控制后续的 llm response 解析
            "messages_list": np.array(messages_list, dtype=object),
            "lm_input_texts": np.array(llm_input_texts, dtype=object), # 用来debug 看日志的
        }

        if prepare_for_update:
            metrics = {}
            for env_output in env_outputs:
                for key, value in env_output["metrics"].items():
                    if key not in metrics:
                        metrics[key] = []
                    metrics[key].append(value)
            mean_metrics = {
                key: np.sum(value) / self.env_nums[key.split("/")[0]]
                for key, value in metrics.items()
            }
            for key, values in metrics.items():
                if not isinstance(values, list):
                    continue
                prefix, suffix = key.split("/", 1)
                non_zero_values = [v for v in values if v != 0]
                if non_zero_values:  # Avoid division by zero
                    non_zero_key = f"{prefix}/non-zero/{suffix}"
                    mean_metrics[non_zero_key] = np.mean(non_zero_values)
            metrics = mean_metrics
            metrics["response_length"] = response_length
            llm_inputs.meta_info = {"metrics": metrics}
        return llm_inputs

    def get_env_inputs(self, lm_outputs: DataProto) -> List[Dict]:
        if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
            responses = self.tokenizer.batch_decode(
                lm_outputs.batch['responses'], 
                skip_special_tokens=True
            )
        else: # dataproto has textual responses
            responses = lm_outputs.non_tensor_batch['response_texts']

        _responses = []
        for response in responses:
            if response.startswith("<think>") or response.startswith("<answer>"):
                _responses.append(response)
            else:
                _responses.append("<think>" + response if self.config.agent_proxy.enable_think else "<answer>" + response) # The LLM generation does not include <think> tags. Add them back here.
        responses = _responses
            
        env_ids = lm_outputs.non_tensor_batch['env_ids']
        env_tags = lm_outputs.non_tensor_batch['env_tags']
        env_inputs = []
        for env_id, env_tag, response in zip(env_ids, env_tags, responses):
            llm_response, actions, answer = self._parse_response(response, env_tag)
            env_inputs.append({
                "env_id": env_id,
                "llm_raw_response": response,
                "llm_response": llm_response,
                "actions": actions,
                "final_answer": answer, # 这个最终 answer 带 <answer></answer> 标签
            })
        return env_inputs

    def formulate_rollouts(self, env_outputs: List[Dict]) -> DataProto:
        llm_inputs = self.get_lm_inputs(env_outputs, prepare_for_update=True)
        return llm_inputs
    
    def get_eval_score(self, lm_outputs: DataProto) -> List[float]:
        """
		grep the score between <score>...</score>
		"""
        def grep_score(text: str) -> float:
            pattern = r'<score>(.*?)</score>'
            score = re.search(pattern, text).group(1)
            if score:
                return float(score)
            else:
                print(f'[DEBUG] score not found in {text}')
                return 0.0

        responses = None
        if lm_outputs.non_tensor_batch is not None and 'response_texts' in lm_outputs.non_tensor_batch.keys():
            responses = lm_outputs.non_tensor_batch['response_texts']
        assert responses is not None, "Responses are not found in the lm_outputs"
        scores = [grep_score(text) for text in responses]
        return scores
    
    def update_eval_score(self, rollouts: DataProto, eval_lm_outputs: DataProto, env_outputs: List[Dict]):
        """
        Update the reward tensor with the LLM reward scores.
        """
        scores = self.get_eval_score(eval_lm_outputs) # 对于每个轨迹最终输出得分
        input_ids = rollouts.batch['input_ids']
        score_tensor = torch.zeros_like(input_ids, dtype=torch.float32)
        score_tensor[:, -1] = torch.tensor(scores, dtype=torch.float32)
        score_tensor = score_tensor[:, 1:] # remove the first token
        normalized_score_tensor = self._normalize_score_tensor(score_tensor, env_outputs)
        rollouts.batch['llm_reward_scores'] = normalized_score_tensor
        return rollouts

    



@hydra.main(version_base = None, config_path = "../../config", config_name = "base")
def main(config):
    import json
    tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
    ctx_manager = ContextManager(config=config, tokenizer=tokenizer)
    print("ctx_manager prefix", ctx_manager.prefix_lookup)
    # batch_list = [
    #     {
    #         "env_ids": 0,
    #         "chat_response": "<think><think></answer> 123. </think><answer> <answer> say | hi </answer></answer>",
    #     },
    #     {
    #         "env_ids": 1,
    #         "chat_response": "<think> 456. </think><answer> 789 </answer><think> 10123 </think><answer> 11111 </answer>",
    #     }
    # ]
    # ctx_manager.action_sep_lookup = {
    #     0: "|",
    #     1: ";"
    # }
    # for item in batch_list:
    #     item["responses"] = tokenizer.encode(item["chat_response"], return_tensors="pt",max_length=512, truncation=True,padding="max_length")[0]
    # batch_dict = collate_fn(batch_list)
    # batch = DataProto.from_single_dict(batch_dict)
    # env_inputs = ctx_manager.get_env_inputs(batch)
    # print(env_inputs)
    


    env_outputs = [
        {
            "env_id": 1,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 1", "reward": 0.5, "actions_left": 2},
                {"state": "###\n#x_#<image>", "llm_response": "Response 2", "reward": 0.8, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 0,
            "metrics": {}
        },
        {
            "env_id": 2,
            "history": [
                {"state": "###\n#x_#<image>", "llm_response": "Response 3", "reward": 0.3, "actions_left": 1},
                {"state": "###\n#x_#<image>", "actions_left": 0}
            ],
            "group_id": 1,
            "metrics": {}
        }
    ]
    
    prefix_lookup = {1: "Initial prompt", 2: "Initial prompt 2"}
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    env_prompt = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False)
    print(env_prompt)
    formulate_rollouts_rst= ctx_manager.formulate_rollouts(env_outputs)
    print(formulate_rollouts_rst)

if __name__ == "__main__":
    main()
    
