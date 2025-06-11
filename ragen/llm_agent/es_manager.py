"""
This is the environment state manager for the LLM agent.
author: Pingyue Zhang
date: 2025-03-30
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import random
import concurrent.futures
import copy
import numpy as np
import time

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
from ragen.llm_agent.observations import BrowserOutputObservation

register_resolvers()

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        self._init_envs()
        self.rollout_cache = None

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

    def _create_env_instance(self, args: tuple):
        """Helper function to create a single environment instance."""
        tag, group_id, env_id = args
        retries = 3
        for attempt in range(retries):
            try:
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                env_obj = REGISTERED_ENVS[env_class](env_config)
                print(f"Init env {env_id} of tag {tag}")
                entry = {'tag': tag, 'group_id': group_id, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
                return entry
            except Exception as e:
                print(f"Error creating env {env_id} (tag: {tag}), attempt {attempt + 1}/{retries}: {e}")
                if attempt + 1 >= retries:
                    raise
                time.sleep(1)

    def _init_env_instances(self, config):
        print("Init envs... env counts: ", sum(config.env_configs.n_groups)* self.group_size)
        env_creation_args = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                group_id = env_id // self.group_size
                env_creation_args.append((tag, group_id, env_id))
            done_groups += n_group
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            env_list = list(executor.map(self._create_env_instance, env_creation_args))
        return env_list

    def _reset_env(self, args):
        """Helper function to reset a single environment instance."""
        entry, seed = args
        retries = 3
        for attempt in range(retries):
            try:
                entry['env'].reset(seed=seed, mode=self.mode)
                status = EnvStatus(seed=seed)
                next_state = self._handle_mm_state(entry['env'].render())
                return entry['env_id'], status, next_state
            except Exception as e:
                print(f"Error resetting env {entry['env_id']} (tag: {entry['tag']}), attempt {attempt + 1}/{retries}: {e}")
                if attempt + 1 < retries:
                    print(f"Recreating env {entry['env_id']} and retrying reset.")
                    try:
                        entry['env'].close()
                    except Exception as close_e:
                        print(f"Error closing failed env {entry['env_id']}: {close_e}")
                    
                    try:
                        new_entry = self._create_env_instance((entry['tag'], entry['group_id'], entry['env_id']))
                        self.envs[entry['env_id']] = new_entry
                        entry = new_entry
                    except Exception as recreate_e:
                        print(f"Failed to recreate env {entry['env_id']}: {recreate_e}")
                        # If recreation fails, we'll just wait and retry reset on the old env instance.
                        time.sleep(1)
                else:
                    raise e
            
    def reset(self, seed: Optional[int] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        # reset all environments
        if self.mode == "train":
            seed = random.randint(0, 1000000) if seed is None else seed # get a random seed
        else:
            seed = 123
        seeds = _expand_seed(seed)
        
        reset_args = zip(envs, seeds)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._reset_env, reset_args))
        
        # update env status and rollout cache
        for env_id, status, next_state in results:
            self.envs[env_id]['status'] = status
            cache = rollout_cache[env_id]
            cache['history'] = self._update_cache_history(
                cache['history'], 
                next_state=next_state, 
                actions_left=self.envs[env_id]['max_actions_per_traj'], 
                num_actions_info=None
            )
            
        self.rollout_cache = rollout_cache
        return rollout_cache

    def _execute_actions(self, entry, actions):
        env = entry['env']
        acc_reward, turn_info, turn_done = 0, {}, False
        executed_actions = []
        retries = 3

        actions_to_process = actions
        is_invalid_action_case = False
        if not actions:
            actions_to_process = ["<invalid_action>"]
            is_invalid_action_case = True

        for action in actions_to_process:
            for attempt in range(retries):
                try:
                    _, reward, done, info = env.step(action)
                    acc_reward += reward
                    turn_info.update(info)
                    if not is_invalid_action_case:
                        executed_actions.append(action)
                    if done:
                        turn_done = True
                    break  # Success, break retry loop
                except Exception as e:
                    print(f"Error stepping env {entry['env_id']} (tag: {entry['tag']}), action '{action}', attempt {attempt + 1}/{retries}: {e}")
                    if attempt + 1 < retries:
                        print(f"Recreating env {entry['env_id']}.")
                        try:
                            env.close()
                        except Exception as close_e:
                            print(f"Error closing failed env {entry['env_id']} during step retry: {close_e}")
                        try:
                            new_entry = self._create_env_instance((entry['tag'], entry['group_id'], entry['env_id']))
                            self.envs[entry['env_id']] = new_entry
                            entry = new_entry
                            env = new_entry['env']
                            
                            turn_info['error'] = f"Environment crashed on action '{action}' and was recreated. Aborting turn."
                            turn_done = True
                            break 
                        except Exception as recreate_e:
                            print(f"Failed to recreate env {entry['env_id']}: {recreate_e}")
                            time.sleep(1) # wait before next retry on old env
                    else:
                        print(f"Failed to step env {entry['env_id']} after {retries} retries. Aborting actions for this turn.")
                        turn_info['error'] = f"Failed to step after {retries} retries: {e}"
                        turn_done = True
            
            if turn_done:
                break
        
        return acc_reward, turn_info, turn_done, executed_actions

    def _log_env_state(self, status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
        obs = self._handle_mm_state(cur_obs)
        status.num_actions += len(executed_actions)
        status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
        actions_left = max_actions_per_traj - status.num_actions # TODO 对于支持多个 action 连续输入的 env, 这里的计算逻辑要修改一下
        if turn_done:
            status.terminated = True # TODO check terminated definition in gymnasium
            status.truncated = not turn_info.get('success', False)
        history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
            'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
            'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
        })
        # filter out invalid actions
        # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
        return status, history
    
    def _step_env(self, env_input):
        """Helper function to step a single environment instance."""
        env_id = env_input['env_id']
        entry = self.envs[env_id]
        # Deepcopy status and history to avoid race conditions
        status = EnvStatus(**vars(entry['status']))
        history = copy.deepcopy(self.rollout_cache[env_id]['history'])

        actions_left_before = entry['max_actions_per_traj'] - status.num_actions

        # execute actions in envs
        valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
        final_answer = env_input.get('final_answer', None)
        if final_answer is not None and final_answer != "":
            acc_reward, turn_info, turn_done, executed_actions = self._execute_actions(entry, [final_answer])
        else:
            acc_reward, turn_info, turn_done, executed_actions = self._execute_actions(entry, valid_actions[:actions_left_before])

        penalty = 0
        if len(valid_actions) != len(env_input['actions']) or not valid_actions:
            penalty = self.sys_config.es_manager.format_penalty
            
        status, history = self._log_env_state(status, history, entry['env'].render(), entry['max_actions_per_traj'], executed_actions, valid_actions, acc_reward, turn_done, turn_info, env_input)
        
        if status.num_actions >= entry['max_actions_per_traj'] and not turn_done:
            status.truncated = True
            status.terminated = True 
            turn_done = True

        return env_id, status, history, penalty, turn_done

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments."""
        env_outputs = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self._step_env, all_env_inputs))

        for env_id, status, history, penalty, turn_done in results:
            self.envs[env_id]['status'] = status
            self.rollout_cache[env_id]['history'] = history
            self.rollout_cache[env_id]["penalty"] += penalty
            
            if not turn_done: # NOTE done environments are not sent for further llm generation (for efficiency)
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    def get_rollout_states(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if k == 'success':
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    custom_metric[k].append(float(v))
            for k, v in custom_metric.items():
                # TODO: Move TURN_LVL_METRICS into the environment
                if "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
                    env_metric[k] = np.sum(v) / (len(cache['history']) - 1) # NOTE: exclude the last observation
                else:
                    env_metric[k] = np.sum(v)


            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            if entry['tag'] == "MetamathQA":
                cache['correct_answer'] = entry['env'].correct_answer
        return rollout_cache

    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        # TODO 这里未来还需要考虑 图文混合的适配
        entry = {} # append state to history
        if isinstance(next_state, str): # text state
            entry['state'] = next_state
        elif isinstance(next_state, BrowserOutputObservation): # 以后和 BrowserOutputObservation 的适配都在这里处理，方便统一控制。
            entry['state'] = str(next_state)
            entry['condensed_state'] = next_state.get_condensed_observation()
            entry['goal'] = next_state.get_goal()
        else: # multimodal state
            entry['state'] = "<images>" * len(next_state) # TODO 这里应该是针对 qwen 多模的适配，可能还不通用。还要考虑多模态输入的适配
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray], BrowserOutputObservation]):
        """Handle the state from the environment
        """
        if isinstance(state, str): # text state
            return state
        elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
            state = [state]
        elif isinstance(state, BrowserOutputObservation): # TODO 这里未来还需要考虑 图文混合的适配
            return state # 还是要在history 中保留原始完整的信息
        elif state is None:
            return "None"
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
	main()
