from .ctx_manager import ContextManager
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import asyncio
import concurrent.futures
import hydra
import os
import re
import time
import numpy as np
from typing import List, Dict, Any
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .base_llm import ConcurrentLLM
from openai.types.chat.chat_completion import ChatCompletion


class VllmWrapperWg: # Thi is a developing class for eval and test
	def __init__(self, config, tokenizer):
		self.config = config
		self.tokenizer = tokenizer
		model_name = config.actor_rollout_ref.model.path
		ro_config = config.actor_rollout_ref.rollout
		self.llm = LLM(
			model_name,
            enable_sleep_mode=True,
            tensor_parallel_size=ro_config.tensor_model_parallel_size,
            dtype=ro_config.dtype,
            enforce_eager=ro_config.enforce_eager,
            gpu_memory_utilization=ro_config.gpu_memory_utilization,
            disable_custom_all_reduce=True,
            disable_mm_preprocessor_cache=True,
            skip_tokenizer_init=False,
            max_model_len=ro_config.max_model_len,
            disable_log_stats=ro_config.disable_log_stats,
            max_num_batched_tokens=ro_config.max_num_batched_tokens,
            enable_chunked_prefill=ro_config.enable_chunked_prefill,
            enable_prefix_caching=True,
			trust_remote_code=True,
		)
		print("LLM initialized")
		self.sampling_params = SamplingParams(
			max_tokens=ro_config.response_length,
			temperature=ro_config.val_kwargs.temperature,
			top_p=ro_config.val_kwargs.top_p,
			top_k=ro_config.val_kwargs.top_k,
			# min_p=0.1,
		)

	def generate_sequences(self, lm_inputs: DataProto):
		"""
		Convert the input ids to text, and then generate the sequences. Finally create a dataproto. 
		This aligns with the verl Worker Group interface.
		"""
		# NOTE: free_cache_engine is not used in the vllm wrapper. Only used in the verl vllm.
		# cache_action = lm_inputs.meta_info.get('cache_action', None)

		input_ids = lm_inputs.batch['input_ids']
		input_texts = self.tokenizer.batch_decode(input_ids, skip_special_tokens=False)
		input_texts = [i.replace("<|endoftext|>", "") for i in input_texts]

		outputs = self.llm.generate(input_texts, sampling_params=self.sampling_params)
		texts = [output.outputs[0].text for output in outputs] 
		lm_outputs = DataProto()
		lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
		lm_outputs.meta_info = lm_inputs.meta_info

		return lm_outputs
	
class ApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        model_info = config.model_info[config.model_config.model_name]
        self.llm_kwargs = model_info.generation_kwargs
        
        
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency
        )
        
        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses, 
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        print(f'[DEBUG] texts: {texts}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs
	

class EvalApiCallingWrapperWg:
    """Wrapper class for API-based LLM calls that fits into the VERL framework"""
    
    def __init__(self, config):
        self.config = config
        self.llm_kwargs = config.generation_kwargs # go to config llm_reward_model_api.generation_kwargs
        model_info = config.api
        self.llm = ConcurrentLLM(
			provider=model_info.provider_name,
            model_name=model_info.model_name,
            max_concurrency=config.model_config.max_concurrency,
			base_url=model_info.base_url,
			api_key=model_info.api_key
        )
        
        print(f'API-based LLM ({model_info.provider_name} - {model_info.model_name}) initialized')


    def generate_sequences(self, lm_inputs: DataProto) -> DataProto:
        """
        Convert the input ids to text, make API calls to generate responses, 
        and create a DataProto with the results.
        """

        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        results, failed_messages = self.llm.run_batch(
            messages_list=messages_list,
            **self.llm_kwargs
        )
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        print(f'[DEBUG] texts[0]: {texts[0]}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

    async def async_generate_sequences(self, lm_inputs: DataProto, mock: bool= False ) -> DataProto:
        """
        Async version of generate_sequences
        """
        messages_list = lm_inputs.non_tensor_batch['messages_list'].tolist()
        if mock:
            lm_outputs = DataProto()
            lm_outputs.non_tensor_batch = {
                'response_texts': ["<score>0.0</score>"] * len(messages_list),
                'env_ids': lm_inputs.non_tensor_batch['env_ids'],
                'group_ids': lm_inputs.non_tensor_batch['group_ids']
                } # this is a bit hard-coded to bypass the __init__ check in DataProto
            lm_outputs.meta_info = lm_inputs.meta_info

            return lm_outputs
        
        # 检测是否在事件循环中，如果是则使用线程池执行同步方法
        try:
            loop = asyncio.get_running_loop()
            # 在事件循环中，使用线程池执行同步的 run_batch
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.llm.run_batch,
                    messages_list=messages_list,
                    **self.llm_kwargs
                )
                results, failed_messages = future.result()
        except RuntimeError:
            # 没有事件循环，直接调用
            results, failed_messages = self.llm.run_batch(
                messages_list=messages_list,
                **self.llm_kwargs
            )
            
        assert not failed_messages, f"Failed to generate responses for the following messages: {failed_messages}"

        texts = [result["response"] for result in results]
        print(f'[DEBUG] texts[0]: {texts[0]}')
        lm_outputs = DataProto()
        lm_outputs.non_tensor_batch = {
			'response_texts': texts,
			'env_ids': lm_inputs.non_tensor_batch['env_ids'],
			'group_ids': lm_inputs.non_tensor_batch['group_ids']
		} # this is a bit hard-coded to bypass the __init__ check in DataProto
        lm_outputs.meta_info = lm_inputs.meta_info
        
        return lm_outputs

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer, async_rollout_manager=None):
		self.config = config
		self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
		self.train_es_manager = EnvStateManager(config, mode="train")
		self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
		self.val_es_manager = EnvStateManager(config, mode="val")
		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer
		self.llm_reward_model_wg = EvalApiCallingWrapperWg(config.llm_reward_model_api) # 用于 llm model based verify 计算
		self.async_rollout_manager = async_rollout_manager # 用于异步 rollout

	def generate_sequences(self, lm_inputs: DataProto):
		# TODO: add kv cache both for the vllm wrapper here and for verl vllm.
		if isinstance(self.actor_wg, RayWorkerGroup):
			padded_lm_inputs, pad_size = pad_dataproto_to_divisor(lm_inputs, self.actor_wg.world_size)
			padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs)
			lm_outputs = unpad_dataproto(padded_lm_outputs, pad_size=pad_size)
			lm_outputs.meta_info = lm_inputs.meta_info
			lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch
		elif isinstance(self.actor_wg, VllmWrapperWg) or isinstance(self.actor_wg, ApiCallingWrapperWg):
			lm_outputs = self.actor_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported actor worker type: {type(self.actor_wg)}")

		return lm_outputs

	async def async_eval_generate_sequences(self, lm_inputs: DataProto):
		if isinstance(self.llm_reward_model_wg, EvalApiCallingWrapperWg):
			lm_outputs = await self.llm_reward_model_wg.async_generate_sequences(lm_inputs, mock=True)
		else:
			raise ValueError(f"Unsupported llm reward model worker type: {type(self.llm_reward_model_wg)}")
		return lm_outputs

	def eval_generate_sequences(self, lm_inputs: DataProto):
		if isinstance(self.llm_reward_model_wg, EvalApiCallingWrapperWg):
			lm_outputs = self.llm_reward_model_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported llm reward model worker type: {type(self.llm_reward_model_wg)}")
		return lm_outputs
	
	async def async_rollout(self, dataproto: DataProto, val=False):
		"""
		异步多轮 rollout 的实现。
		每个环境的 rollout 独立进行，并行进行。不在通过 config.agent_proxy.max_turn 环境交互的步频。
		外层直接循环 env_outputs, 提交 interact loop 任务逻辑。
		interact loop 中包含完整的 rollout 步骤，包括：
		1. 获取当前轮次的 ctx_manager.get_lm_inputs
		2. 生成响应 generate_sequences
		3. 生成环境输入 ctx_manager.get_env_inputs
		4. 环境交互 es_manager.step
		以上循环直到满足环境退出条件
		最后合并跟新完整的 rollout 状态。
		"""

		async def callback(completions: ChatCompletion, info: Dict[str, Any], exception: Exception):
			assert exception is None, f"exception: {exception}"
			env_id = info["env_id"]
			messages = info["messages"]
			message = completions.choices[0].message
			messages.append({"role": message.role, "content": message.content})
			if env_id == 0:
				print(f'[DEBUG] callback for env_id: {env_id} add new response: {message.content}')
		
		async def interact_loop_callback(env_id, env_output, es_manager, ctx_manager, max_turn):
			if env_id == 0:
				print(f'[DEBUG] Start Async Rollout for env_id: {env_id}')
			for i in range(max_turn): # 外部限制一下交互环境的最大轮次
				if env_id == 0:
					print(f'[DEBUG] Async Rollout for env_id: {env_id} turn {i}')
				lm_inputs: DataProto = ctx_manager.get_lm_inputs([env_output], prepare_for_update=False)
				lm_inputs.meta_info = dataproto.meta_info
				
				messages = lm_inputs.non_tensor_batch["messages_list"].tolist()[0]
				pre_len = len(messages)
				model_name = "/".join(self.config.actor_rollout_ref.model.path.split("/")[-2:])
				if env_id == 0:
					print(f'[DEBUG] callback for env_id: {env_id} messages: {messages}')
					print(f'[DEBUG] callback for env_id: {env_id} model_name: {model_name}')
				await self.async_rollout_manager.chat_scheduler.submit_chat_completions(
					callback=callback,
					callback_additional_info={"env_id": env_id, "messages": messages},
					model=model_name, # https://platform.openai.com/docs/api-reference/chat/create 这里往下都是 chat_complete_request 的参数
					messages=messages,
					max_completion_tokens=self.config.actor_rollout_ref.rollout.response_length,
					temperature=self.config.actor_rollout_ref.rollout.temperature,
				)
				assert len(messages) == pre_len + 1, f"messages length: {len(messages)} != pre_len: {pre_len} + 1"
				lm_outputs = DataProto()
				lm_outputs.meta_info = lm_inputs.meta_info
				lm_outputs.non_tensor_batch = lm_inputs.non_tensor_batch.copy()
				lm_outputs.non_tensor_batch['response_texts'] = np.array([messages[-1]["content"]], dtype=object)
				
				env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
				env_id_result, history, penalty, turn_done = await es_manager.async_step_by_env_id(env_inputs[0]['env_id'], env_inputs[0], timeout=60.0)
				assert env_id_result == env_id, f"env_id_result: {env_id_result} != env_id: {env_id}"

				es_manager.rollout_cache[env_id_result]['history'] = history
				es_manager.rollout_cache[env_id_result]["penalty"] += penalty

				if turn_done:
					print(f'[DEBUG] env_id: {env_id} finished at turn: {i+1}')
					break

				env_output = es_manager.rollout_cache[env_id]
	
		s_time = time.time()
		print(f"[DEBUG] Start Async Rollout.")
		max_turn = self.config.agent_proxy.max_turn
		es_manager = self.val_es_manager if val else self.train_es_manager
		ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
		env_outputs = es_manager.reset()
		rollout_tasks = []
		for env_output in env_outputs:
			rollout_tasks.append(asyncio.create_task(interact_loop_callback(env_output['env_id'], env_output, es_manager, ctx_manager, max_turn)))
		await asyncio.gather(*rollout_tasks)
		print(f'[DEBUG] Async Rollout tasks finished.')
		e_time = time.time()
		print(f'[DEBUG] Async Rollout tasks finished in {e_time - s_time} seconds.')
		
		rollout_states = es_manager.get_rollout_states() 
		rollouts = ctx_manager.formulate_rollouts(rollout_states)
		# self.tokenizer.batch_decode(rollouts.batch['input_ids'], skip_special_tokens=False) # see all the trajectories
		if self.llm_reward_model_wg:
			# 对结果做奖励打分，score 作为 sequence level 的 score 放到最后
			eval_lm_inputs: DataProto = ctx_manager.get_eval_lm_inputs(rollout_states)
			eval_lm_outputs: DataProto = await self.async_eval_generate_sequences(eval_lm_inputs)
			rollouts = ctx_manager.update_eval_score(rollouts, eval_lm_outputs, rollout_states)

		metrics, valid_action_count, invalid_action_count = self.get_stats_from_rollout_states(rollout_states)
		print(f'[DEBUG] rollout_states metrics: {metrics}')
		print(f'[DEBUG] valid_action_count: {valid_action_count}')
		print(f'[DEBUG] invalid_action_count: {invalid_action_count}')
		rollouts.meta_info["metrics"].update(metrics)
		return rollouts


	def rollout(self, dataproto: DataProto, val=False):
		s_time = time.time()
		print(f"[DEBUG] Start Sync Rollout.")
		es_manager = self.val_es_manager if val else self.train_es_manager
		ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
		env_outputs = es_manager.reset()

		for i in range(self.config.agent_proxy.max_turn):
			print(f'[DEBUG] rollout turn {i}')
			lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False) # 获取当前轮次的 prefill prompts
			lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop when max length is reached. make sure this can be done
			# print(f'[DEBUG] rollout turn {i} longest lm_inputs: {sorted(lm_inputs.non_tensor_batch["lm_input_texts"], key=len)[-1]}')
			print(f'[DEBUG] rollout turn {i} lm_inputs[0]: {lm_inputs.non_tensor_batch["lm_input_texts"][0]}')
			lm_outputs: DataProto = self.generate_sequences(lm_inputs)

			if lm_outputs.batch is not None and 'responses' in lm_outputs.batch.keys():
				responses = self.tokenizer.batch_decode(
					lm_outputs.batch['responses'], 
					skip_special_tokens=True
				)
			else: # dataproto has textual responses
				responses = lm_outputs.non_tensor_batch['response_texts']
			print(f'[DEBUG] rollout turn {i} responses[0]: {responses[0]}')
			
			env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
			env_outputs: List[Dict] = es_manager.step(env_inputs)
			if len(env_outputs) == 0: # all finished
				break
		print(f'[DEBUG] Sync Rollout tasks finished.')
		e_time = time.time()
		print(f'[DEBUG] Sync Rollout tasks finished in {e_time - s_time} seconds.')
		
		rollout_states = es_manager.get_rollout_states() 
		rollouts = ctx_manager.formulate_rollouts(rollout_states)
		# self.tokenizer.batch_decode(rollouts.batch['input_ids'], skip_special_tokens=False) # see all the trajectories
		if self.llm_reward_model_wg:
			# 对结果做奖励打分，score 作为 sequence level 的 score 放到最后
			# TODO: 异步提效。
			eval_lm_inputs: DataProto = ctx_manager.get_eval_lm_inputs(rollout_states)
			eval_lm_outputs: DataProto = self.eval_generate_sequences(eval_lm_inputs)
			rollouts = ctx_manager.update_eval_score(rollouts, eval_lm_outputs, rollout_states)

		metrics, valid_action_count, invalid_action_count = self.get_stats_from_rollout_states(rollout_states)
		print(f'[DEBUG] rollout_states metrics: {metrics}')
		print(f'[DEBUG] valid_action_count: {valid_action_count}')
		print(f'[DEBUG] invalid_action_count: {invalid_action_count}')
		rollouts.meta_info["metrics"].update(metrics)
		return rollouts
	
	def get_stats_from_rollout_states(self, rollout_states: List[Dict]):
		"""
		统计全部 rollout 轨迹指标
		轨迹长度：mean，max, min
		输出 answer 率
		输出正确 action 数
		输出错误 action 数
		输出无法解析 action 数
		rollout_states: [{
			"env_id": self.env_id,
            "history": self.history,
            "group_id": self.group_id,
            "tag": self.tag,
            "penalty": 0, 
            "metrics": env_metric,
		}
		]
		"""
		trajectory_lengths = []
		reach_answer = []
		correct_action = []
		incorrect_action = []
		unparseable_action = []
		print(f'[DEBUG] rollout_states[0] last history: {rollout_states[0]["history"][-1]}')
		for rollout_state in rollout_states:
			trajectory_lengths.append(len(rollout_state["history"]))
			
			if rollout_state["history"][-1].get("meta_info", {}).get("status", "") == "answer_output":
				reach_answer.append(1)
			else:
				reach_answer.append(0)

			for entry in rollout_state["history"]:
				if entry.get('info', {}).get("meta_info", {}).get("status", "") == "action_output":
					if entry.get('info', {}).get("meta_info", {}).get("valid_action", None):
						correct_action.append(entry.get('info', {}).get("meta_info", {}).get("valid_action", []) )
				else:
					if entry.get('info', {}).get("meta_info", {}).get("invalid_action", None):
						incorrect_action.append(entry.get('info', {}).get("meta_info", {}).get("invalid_action", []))
				if entry.get('info', {}).get("meta_info", {}).get("status", "") == "action_error":
					unparseable_action.append(1)
		print(f'[DEBUG] correct_action: {correct_action}')
		print(f'[DEBUG] incorrect_action: {incorrect_action}')
		print(f'[DEBUG] unparseable_action: {unparseable_action}')
		valid_action_count, invalid_action_count = {}, {}
		for action in correct_action:
			for action_item in action:
				valid_action_count[action_item] = valid_action_count.get(action_item, 0) + 1
		for action in incorrect_action:
			for action_item in action:
				invalid_action_count[action_item] = invalid_action_count.get(action_item, 0) + 1

		return {
			"rollout/mean_trajectory_lengths": sum(trajectory_lengths) / len(trajectory_lengths),
			"rollout/max_trajectory_lengths": max(trajectory_lengths),
			"rollout/min_trajectory_lengths": min(trajectory_lengths),
			"rollout/answer_ratio": sum(reach_answer) / len(reach_answer),
			"rollout/answer_count": sum(reach_answer),
			"rollout/correct_action_count": sum([len(action) for action in correct_action]),
			"rollout/incorrect_action_count": sum([len(action) for action in incorrect_action]),
			"rollout/unparseable_action_count": sum(unparseable_action),
		}, valid_action_count, invalid_action_count



@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
	os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(config.system.CUDA_VISIBLE_DEVICES)
	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
	actor_wg = VllmWrapperWg(config, tokenizer)
	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
	import time
	for _ in range(3):
		start_time = time.time()
		rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample':config.actor_rollout_ref.rollout.do_sample, 'validate': True}), val=True)
		end_time = time.time()
		print(f'rollout time: {end_time - start_time} seconds')
		# print rollout rewards from the rm_scores
		rm_scores = rollouts.batch["rm_scores"]
		metrics = rollouts.meta_info["metrics"]
		avg_reward = rm_scores.sum(-1).mean().item()
		print(f'rollout rewards: {avg_reward}')
		print(f'metrics:')
		for k, v in metrics.items():
			print(f'{k}: {v}')

# @hydra.main(version_base=None, config_path="../../config", config_name="evaluate_api_llm")
# def main(config):
# 	# detect config name from python -m ragen.llm_agent.agent_proxy --config_name frozen_lake
# 	tokenizer = AutoTokenizer.from_pretrained(config.actor_rollout_ref.model.path)
# 	actor_wg = ApiCallingWrapperWg(config, tokenizer)
# 	proxy = LLMAgentProxy(config, actor_wg, tokenizer)
# 	import time
# 	start_time = time.time()
# 	rollouts = proxy.rollout(DataProto(batch=None, non_tensor_batch=None, meta_info={'eos_token_id': 151645, 'pad_token_id': 151643, 'recompute_log_prob': False, 'do_sample': False, 'validate': True}), val=True)
# 	print(f'[DEBUG] rollouts: {rollouts}')
# 	end_time = time.time()
# 	print(f'rollout time: {end_time - start_time} seconds')
# 	# print rollout rewards from the rm_scores
# 	rm_scores = rollouts.batch["rm_scores"]
# 	metrics = rollouts.meta_info["metrics"]
# 	avg_reward = rm_scores.sum(-1).mean().item()
# 	print(f'rollout rewards: {avg_reward}')
# 	print(f'metrics:')
# 	for k, v in metrics.items():
# 		print(f'{k}: {v}')



if __name__ == "__main__":
	main()
