from .ctx_manager import ContextManager
from .es_manager import EnvStateManager
from vllm import LLM, SamplingParams
from verl.single_controller.ray.base import RayWorkerGroup
from transformers import AutoTokenizer, AutoModelForCausalLM
from verl import DataProto
import hydra
import os
import re
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from .base_llm import ConcurrentLLM


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
        self.llm_kwargs = config.generation_kwargs
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

class LLMAgentProxy:
	"""
	The proxy means the llm agent is trying to generate some rollout **at this time**, **at this model state**, **at this env state from the env config**
	"""
	def __init__(self, config, actor_rollout_wg, tokenizer):
		self.config = config
		self.train_ctx_manager = ContextManager(config, tokenizer, mode="train")
		self.train_es_manager = EnvStateManager(config, mode="train")
		self.val_ctx_manager = ContextManager(config, tokenizer, mode="val")
		self.val_es_manager = EnvStateManager(config, mode="val")
		self.actor_wg = actor_rollout_wg
		self.tokenizer = tokenizer
		self.llm_reward_model_wg = EvalApiCallingWrapperWg(config.llm_reward_model_api) # 用于 llm model based verify 计算

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

	def eval_generate_sequences(self, lm_inputs: DataProto):
		if isinstance(self.llm_reward_model_wg, EvalApiCallingWrapperWg):
			lm_outputs = self.llm_reward_model_wg.generate_sequences(lm_inputs)
		else:
			raise ValueError(f"Unsupported llm reward model worker type: {type(self.llm_reward_model_wg)}")
		return lm_outputs

	def rollout(self, dataproto: DataProto, val=False):
		es_manager = self.val_es_manager if val else self.train_es_manager
		ctx_manager = self.val_ctx_manager if val else self.train_ctx_manager
		env_outputs = es_manager.reset()

		"""
		# TODO 这里也是同步锁定了，考虑用异步流水线提升效率吧。
		虽然verl 实现了 rollout 阶段的异步流水线，但是agent 这里需要多步交互，rollout 阶段存在切换，所以需要实现两层意义上的异步流水线：1. Env.step 异步并发，2. 上面generate_sequences 中 padded_lm_outputs = self.actor_wg.generate_sequences(padded_lm_inputs) 也要换成 self.async_rollout_manager.generate_sequences(gen_batch) 提升推理截断的异步流水线。
		if not self.async_rollout_mode:
			gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
		else:
			self.async_rollout_manager.wake_up()
			gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
			self.async_rollout_manager.sleep()
		"""
		for i in range(self.config.agent_proxy.max_turn):
			print(f'[DEBUG] rollout turn {i}')
			lm_inputs: DataProto = ctx_manager.get_lm_inputs(env_outputs, prepare_for_update=False) # 获取当前轮次的 prefill prompts
			lm_inputs.meta_info = dataproto.meta_info # TODO: setup vllm early stop when max length is reached. make sure this can be done
			# print(f'[DEBUG] rollout turn {i} longest lm_inputs: {sorted(lm_inputs.non_tensor_batch["lm_input_texts"], key=len)[-1]}')
			print(f'[DEBUG] rollout turn {i} lm_inputs[0]: {lm_inputs.non_tensor_batch["lm_input_texts"][0]}')
			lm_outputs: DataProto = self.generate_sequences(lm_inputs)
			env_inputs: List[Dict] = ctx_manager.get_env_inputs(lm_outputs)
			# TODO: env 里面可以添加 env input 打点统计，统计，每个环境组的，解析正确率，执行步长，是否抵达 answer 输出，页面跳转轨迹等信息，辅助分析环境组内，模型执行的稳定性和行为表现。
			env_outputs: List[Dict] = es_manager.step(env_inputs)
			if len(env_outputs) == 0: # all finished
				break
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
		for rollout_state in rollout_states:
			trajectory_lengths.append(len(rollout_state["history"]))
			print(f'[DEBUG] rollout_state last history: {rollout_state["history"][-1]}')
			if rollout_state["history"][-1].get("meta_info", {}).get("status", "") == "answer_output":
				reach_answer.append(1)
			else:
				reach_answer.append(0)

			for entry in rollout_state["history"]:
				if entry.get('info', {}).get("meta_info", {}).get("status", "") == "action_output":
					correct_action.append(entry.get('info', {}).get("meta_info", {}).get("valid_action", []))
				else:
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
