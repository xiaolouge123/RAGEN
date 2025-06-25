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
import ray

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

@ray.remote(num_cpus=1)
class EnvActor:
    """An actor that manages a single environment instance."""
    def __init__(self, tag: str, group_id: int, env_id: int, sys_config: Any, mode: str):
        self.tag = tag
        self.group_id = group_id
        self.env_id = env_id
        self.sys_config = sys_config
        self.mode = mode
        self.status = EnvStatus()
        self.history = []
        
        self._create_env_instance()

    def _create_env_instance(self):
        """Creates a single environment instance within the actor."""
        retries = 3
        for attempt in range(retries):
            try:
                cfg_template = self.sys_config.custom_envs[self.tag]
                env_class = cfg_template.env_type
                self.max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                self.env = REGISTERED_ENVS[env_class](env_config)
                print(f"Init env {self.env_id} of tag {self.tag} in actor.")
                return
            except Exception as e:
                print(f"Error creating env {self.env_id} (tag: {self.tag}) in actor, attempt {attempt + 1}/{retries}: {e}")
                if attempt + 1 >= retries:
                    raise
                time.sleep(1)

    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray], BrowserOutputObservation]):
        if isinstance(state, str):
            return state
        elif isinstance(state, np.ndarray):
            return [PIL.Image.fromarray(state, mode='RGB')]
        elif isinstance(state, BrowserOutputObservation):
            return state
        elif state is None:
            return "None"
        return [PIL.Image.fromarray(_state, mode='RGB') for _state in state]

    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        if num_actions_info is not None:
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {}
        if isinstance(next_state, str):
            entry['state'] = next_state
        elif isinstance(next_state, BrowserOutputObservation):
            entry['state'] = str(next_state)
            entry['condensed_state'] = next_state.get_condensed_observation()
            entry['goal'] = next_state.get_goal()
            entry['gt'] = next_state.get_gt()
            entry['final_answer'] = next_state.get_answer()
        else:
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _execute_actions(self, actions):
        acc_reward, turn_info, turn_done = 0, {}, False
        executed_actions = []
        retries = 3

        actions_to_process = actions or ["<invalid_action>"]
        is_invalid_action_case = not actions

        for action in actions_to_process:
            for attempt in range(retries):
                try:
                    _, reward, done, info = self.env.step(action)
                    acc_reward += reward
                    turn_info.update(info)
                    if not is_invalid_action_case:
                        executed_actions.append(action)
                    if done:
                        turn_done = True
                    break
                except Exception as e:
                    print(f"Error stepping env {self.env_id} (tag: {self.tag}), action '{action}', attempt {attempt + 1}/{retries}: {e}")
                    # Simplified retry, just wait and retry. In a real scenario, might recreate env.
                    if attempt + 1 >= retries:
                        turn_info['error'] = f"Failed to step after {retries} retries: {e}"
                        turn_done = True
                    time.sleep(1)
            
            if turn_done:
                break
        
        return acc_reward, turn_info, turn_done, executed_actions

    def reset(self, seed: Optional[int] = None):
        """Resets the environment and returns the initial state."""
        retries = 3
        for attempt in range(retries):
            try:
                self.env.reset(seed=seed, mode=self.mode)
                self.status = EnvStatus(seed=seed)
                self.history = []
                next_state = self._handle_mm_state(self.env.render())
                self.history = self._update_cache_history(
                    self.history, 
                    next_state=next_state, 
                    actions_left=self.max_actions_per_traj, 
                    num_actions_info=None
                )
                return self.env_id, self.history, self.group_id, self.tag
            except Exception as e:
                print(f"Error resetting env {self.env_id} (tag: {self.tag}) in actor, attempt {attempt + 1}/{retries}: {e}")
                if attempt + 1 >= retries:
                    raise
                self._create_env_instance() # Try to recreate the env
        
    def step(self, env_input: Dict):
        """Steps the environment with the given actions."""
        actions = env_input['actions']
        
        actions_left_before = self.max_actions_per_traj - self.status.num_actions

        valid_actions = self._extract_map_valid_actions(env_input, actions)
        final_answer = env_input.get('final_answer', None)

        if final_answer is not None and final_answer != "":
            # 已经输出答案了，就不用再执行了，但是还是要 env 记录下 answer 结果到 history 中。已经走到输出结果的阶段了。
            acc_reward, turn_info, turn_done, executed_actions = self._execute_actions([final_answer])  # 这个最终 answer 带 <answer></answer> 标签
        else:
            acc_reward, turn_info, turn_done, executed_actions = self._execute_actions(valid_actions[:actions_left_before])
        
        penalty = 0
        if len(valid_actions) != len(actions) or not valid_actions:
            penalty = self.sys_config.es_manager.format_penalty
        
        # Log state
        obs = self._handle_mm_state(self.env.render())
        self.status.num_actions += len(executed_actions)
        self.status.rewards.append(acc_reward)
        actions_left = self.max_actions_per_traj - self.status.num_actions
        if turn_done:
            self.status.terminated = True
            self.status.truncated = not turn_info.get('success', False)

        self.history = self._update_cache_history(self.history, next_state=obs, actions_left=actions_left, num_actions_info={
            'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
            'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
        })

        if self.status.num_actions >= self.max_actions_per_traj and not turn_done:
            self.status.truncated = True
            self.status.terminated = True
            turn_done = True

        return self.env_id, self.history, penalty, turn_done
    
    def get_rollout_state(self):
        """Returns the final collected trajectory and metrics."""
        env_metric = {
            'success': float(self.status.terminated and (not self.status.truncated)),
            'num_actions': self.status.num_actions,
        }
        custom_metric = {}
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        exclude_key = ['meta_info']

        for turn in self.history:
            for k, v in turn.get('info', {}).items():
                if k == 'success': continue
                if k in exclude_key: continue
                if k not in custom_metric: custom_metric[k] = []
                custom_metric[k].append(float(v))

        for k, v in custom_metric.items():
            if "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
                env_metric[k] = np.sum(v) / (len(self.history) - 1) if len(self.history) > 1 else 0
            else:
                env_metric[k] = np.sum(v)
        
        if self.history:
            self.history[-1]['metrics'] = custom_metric
        
        env_metric = {f"{self.tag}/{k}": v for k, v in env_metric.items()}

        final_state = {
            "env_id": self.env_id,
            "history": self.history,
            "group_id": self.group_id,
            "tag": self.tag,
            "penalty": 0, # penalty is handled in EnvStateManager
            "metrics": env_metric,
        }
        if self.tag == "MetamathQA":
             final_state['correct_answer'] = self.env.correct_answer
        
        return final_state

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(self.env.config, 'action_lookup', None)
        if action_lookup is None:
            return actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            return [rev_action_lookup[action] for action in actions if action in rev_action_lookup]

    def close(self):
        self.env.close()

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
        """Initialize the environments as Ray actors."""
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        
        print(f"Initializing {self.env_groups * self.group_size} environment actors...")
        env_creation_args = []
        done_groups = 0
        for tag, n_group in zip(self.config.env_configs.tags, self.config.env_configs.n_groups):
            for env_idx in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                group_id = env_idx // self.group_size
                env_creation_args.append((tag, group_id, env_idx))
            done_groups += n_group
        
        # Each actor gets 1 CPU.
        self.env_actors = [EnvActor.remote(tag, group_id, env_id, self.sys_config, self.mode) for tag, group_id, env_id in env_creation_args]
        print("All environment actors created.")

    def _create_env_instance(self, args: tuple):
        """Helper function to create a single environment instance."""
        # This function is now part of the EnvActor. Keeping it here for reference or potential future single-process use.
        pass

    def _init_env_instances(self, config):
        # This function is now replaced by _init_envs.
        pass

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        def _expand_seed(seed: int):
            # Each group gets a different seed, but all envs in the same group start with the same seed.
            seeds = [[seed + i] for i in range(self.env_groups)] 
            return sum([s * self.group_size for s in seeds], [])

        if self.mode == "train":
            seed = random.randint(0, 1000000) if seed is None else seed
        else:
            seed = 123
        seeds = _expand_seed(seed)

        # Reset all environment actors in parallel
        print("Resetting all environment actors...")
        reset_futures = [actor.reset.remote(seed=s) for actor, s in zip(self.env_actors, seeds)]
        results = ray.get(reset_futures)
        print("All environment actors have been reset.")
        
        # Initialize rollout cache from actor results
        self.rollout_cache = [{} for _ in range(len(self.env_actors))]
        env_outputs = []
        for env_id, history, group_id, tag in results:
            cache_entry = {"env_id": env_id, "history": history, "group_id": group_id, "tag": tag, "penalty": 0}
            self.rollout_cache[env_id] = cache_entry
            env_outputs.append(cache_entry)
            
        return env_outputs

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments by calling remote actors."""
        
        # Create a mapping from env_id to input
        env_input_map = {inp['env_id']: inp for inp in all_env_inputs}

        # Select actors that need to be stepped, using the env_id from the remote actor object
        active_actors_with_inputs = [
            (self.env_actors[env_id], input_data) 
            for env_id, input_data in env_input_map.items()
        ]
        
        step_futures = [actor.step.remote(env_input) for actor, env_input in active_actors_with_inputs]
        
        if not step_futures:
            return []

        results = ray.get(step_futures)

        env_outputs = []
        for env_id, history, penalty, turn_done in results:
            self.rollout_cache[env_id]['history'] = history
            self.rollout_cache[env_id]["penalty"] += penalty
            
            if not turn_done:
                # Only include environments that are not done
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    def get_rollout_states(self):
        """Get the final output for all environment actors by calling them in parallel."""
        print("Gathering rollout states from all environment actors...")
        get_state_futures = [actor.get_rollout_state.remote() for actor in self.env_actors]
        rollout_states = ray.get(get_state_futures)
        print("All rollout states gathered.")
        
        # The actor's get_rollout_state method now computes the metrics, so we just return the result.
        # We need to re-apply the penalties that were accumulated in the manager.
        for state in rollout_states:
            state['penalty'] = self.rollout_cache[state['env_id']].get('penalty', 0)

        return rollout_states

    def close(self):
        """Close all environment actors."""
        print("Closing all environment actors...")
        close_futures = [actor.close.remote() for actor in self.env_actors]
        try:
            ray.get(close_futures, timeout=60)
            print("All environment actors closed gracefully.")
        except ray.exceptions.RayTaskError as e:
            print(f"An error occurred while closing actors: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during close: {e}")
        
    def render(self):
        # This would require fetching state from all actors, which is slow.
        # It's better to implement a 'render.remote()' on the actor if needed for debugging specific instances.
        print("Render is not supported in distributed mode. Call 'render.remote()' on a specific actor handle.")
        return []

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
