import io
import time
import uuid
import base64
import atexit
import random
import tenacity
import html2text
import multiprocessing
import datasets
import numpy as np
from PIL import Image
from typing import Optional, Any, Dict
from dataclasses import dataclass, field
# import re
# import redis
# import pickle

# register openended gym environments
import browsergym.core
import gymnasium as gym
from browsergym.utils.obs import flatten_dom_to_str, overlay_som

from ragen.log import openhands_logger as logger
from ragen.env.webbrowser.utils import stop_if_should_exit, should_continue, should_exit
from ragen.env.webbrowser.exception import BrowserInitException
from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.webbrowser.config import WebBrowserEnvConfig
from ragen.utils import all_seed
from ragen.llm_agent.observations import BrowserOutputObservation


# class WebPageCacheClient:
#     """
#     A Redis client for caching webpage observations.
#     The cache key is the page URL, and the value is the observation result.
#     """
#     def __init__(self, host='localhost', port=6379, db=0, expiration_time=86400):
#         """
#         Initializes the Redis client.
#         Args:
#             host (str): Redis server host.
#             port (int): Redis server port.
#             db (int): Redis database number.
#             expiration_time (int): Cache expiration time in seconds. Defaults to 24 hours.
#         """
#         try:
#             # Add a timeout to avoid blocking forever if Redis is slow
#             self.redis_client = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=2)
#             self.redis_client.ping()
#             logger.info("Successfully connected to Redis for page caching.")
#         except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
#             logger.warning(f"Could not connect to Redis for page caching: {e}. Cache will be disabled.")
#             self.redis_client = None
#         self.expiration_time = expiration_time

#     def get(self, url: str) -> Optional[Dict]:
#         """
#         Retrieves a cached observation for a given URL.
#         Args:
#             url (str): The URL to retrieve from the cache.
#         Returns:
#             The cached observation dictionary, or None if not found or if Redis is unavailable.
#         """
#         if not self.redis_client:
#             return None
        
#         try:
#             cached_data = self.redis_client.get(url)
#             if cached_data:
#                 logger.info(f"Cache hit for URL: {url}")
#                 return pickle.loads(cached_data)
#             else:
#                 logger.info(f"Cache miss for URL: {url}")
#                 return None
#         except Exception as e:
#             logger.error(f"Error getting cache from Redis for {url}: {e}")
#             return None

#     def set(self, url: str, observation: Dict):
#         """
#         Caches an observation for a given URL.
#         Args:
#             url (str): The URL to use as the cache key.
#             observation (dict): The observation object to cache.
#         """
#         if not self.redis_client:
#             return
        
#         try:
#             serialized_data = pickle.dumps(observation)
#             self.redis_client.setex(url, self.expiration_time, serialized_data)
#             logger.info(f"Cached observation for URL: {url}")
#         except Exception as e:
#             logger.error(f"Error setting cache to Redis for {url}: {e}")


@dataclass
class Task:
    task_idx: int
    data_source: str
    data_url: str
    instruction: str
    action_tip: str
    ground_truth: str
            
    def get_task_goal(self):
        if self.action_tip:
            return f"在{self.data_url}网站中，找到{self.instruction}，结果输入result.md文件，网站中寻找到目标数据的经验tips总结如下: {self.action_tip}"
        else:
            return f"在{self.data_url}网站中，找到{self.instruction}，结果输入result.md文件"


def format_browser_observation(obs: dict):
    # logger.info(f" axtree_object:  {obs.get('axtree_object', {})}")
    # logger.info(f" extra_element_properties:  {obs.get('extra_element_properties', {})}")
    return {
        'content': obs['text_content'],
        'url': obs.get('url', ''),
        'screenshot': obs.get('screenshot', None),
        'set_of_marks': obs.get('set_of_marks', None),
        'goal_image_urls': obs.get('image_content', []),
        'open_pages_urls': obs.get('open_pages_urls', []),
        'active_page_index': obs.get('active_page_index', -1),
        'dom_object': obs.get('dom_object', {}),
        'axtree_object': obs.get('axtree_object', {}),
        'extra_element_properties': obs.get('extra_element_properties', {}),
        'focused_element_bid': obs.get('focused_element_bid', None),
        'last_browser_action': obs.get('last_action', ''),
        'last_browser_action_error': obs.get('last_action_error', ''),
        'error': True if obs.get('last_action_error', '') else False,
        'trigger_by_action': 'browse_interactive',
    }


class WebBrowserEnv(BaseLanguageBasedEnv):
    def __init__(self, config: Optional[WebBrowserEnvConfig] = None, **kwargs: any) -> None:
        super().__init__()
        self.html_text_converter = self.get_html_text_converter()
        self.config = config if config is not None else WebBrowserEnvConfig()
        self.train_data = datasets.load_dataset("parquet", data_files=self.config.train_path)
        self.val_data = datasets.load_dataset("parquet", data_files=self.config.val_path)
        self.browser_retries_limit = getattr(self.config, "browser_retries_limit", 3)

        # Initialize browser environment process
        multiprocessing.set_start_method('spawn', force=True)
        self.browser_side, self.agent_side = multiprocessing.Pipe()

        self.init_browser()
        atexit.register(self.close)

        self.render_cache: BrowserOutputObservation = None
        self.current_task: Task = None
        self.browser_retries = 0

    def get_html_text_converter(self):
        html_text_converter = html2text.HTML2Text()
        # ignore links and images
        html_text_converter.ignore_links = False
        html_text_converter.ignore_images = True
        # use alt text for images
        html_text_converter.images_to_alt = True
        # disable auto text wrapping
        html_text_converter.body_width = 0
        return html_text_converter
    
    @tenacity.retry(
        wait=tenacity.wait_fixed(1),
        stop=tenacity.stop_after_attempt(5) | stop_if_should_exit(),
        retry=tenacity.retry_if_exception_type(BrowserInitException),
    )
    def init_browser(self):
        logger.debug('Starting browser env...')
        try:
            self.process = multiprocessing.Process(target=self.browser_process)
            self.process.start()
        except Exception as e:
            logger.error(f'Failed to start browser process: {e}')
            raise

        if not self.check_alive(timeout=200):
            self.close()
            raise BrowserInitException('Failed to start browser environment.')

    def browser_process(self):
        # cache_client = WebPageCacheClient()
        env = gym.make(
            'browsergym/openended',
            task_kwargs={'start_url': 'about:blank', 'goal': 'PLACEHOLDER_GOAL'},
            wait_for_user_message=False,
            headless=True,
            disable_env_checker=True,
            tags_to_mark='all',
        )
        obs, info = env.reset() # 这个环境在 browsergym.core.env 中定义 BrowserEnv.reset
        self.render_cache = obs
        logger.info('Successfully called env.reset')
        
        logger.info('Browser env started.')

        while should_continue():
            try:
                if self.browser_side.poll(timeout=0.01):
                    unique_request_id, action_data = self.browser_side.recv()

                    # shutdown the browser environment
                    if unique_request_id == 'SHUTDOWN':
                        logger.debug('SHUTDOWN recv, shutting down browser env...')
                        env.close()
                        return
                    elif unique_request_id == 'IS_ALIVE':
                        self.browser_side.send(('ALIVE', None))
                        continue
                    elif unique_request_id == 'RESET':
                        # reset 后也没必要给外面看空白页
                        obs, info = env.reset()
                        # TODO 考虑封装一下
                        html_str = flatten_dom_to_str(obs['dom_object']) # 
                        obs['text_content'] = self.html_text_converter.handle(html_str)
                        # make observation serializable
                        obs['set_of_marks'] = self.image_to_png_base64_url(
                            overlay_som(
                                obs['screenshot'], obs.get('extra_element_properties', {})
                            ),
                            add_data_prefix=True,
                        )
                        obs['screenshot'] = self.image_to_png_base64_url(
                            obs['screenshot'], add_data_prefix=True
                        )
                        obs['active_page_index'] = obs['active_page_index'].item()
                        obs['elapsed_time'] = obs['elapsed_time'].item()
                        self.render_cache = obs
                        self.browser_side.send(('RESET_DONE', self.render_cache)) # send back the init status
                        continue

                    action = action_data['action']

                    # Caching Logic for 'nav' action set: goto, go_back, go_forward
                    url = None
                    if "goto(" in action or "go_back(" in action or "go_forward(" in action:
                        # TODO 这里的缓存实现要满足两层需要，一个是给 agent 看的 observation 层，这个比较好实现，只要缓存 url 和 obs 内容即可，但还有一层，需要 playwright 的浏览器环境也能继承这个缓存的 session，显然这个我还不知道咋怎么实现，所以先不考虑缓存的问题。
                        pass 
                    
                    obs, reward, terminated, truncated, info = env.step(action)
                    # obs 包含如下字段： chat_messages,goal,goal_object, open_pages_urls,open_pages_titles,active_page_index,url,screenshot,dom_object,axtree_object,extra_element_properties,focused_element_bid,last_action,last_action_error,elapsed_time

                    # TODO 考虑封装一下
                    # add text content of the page
                    html_str = flatten_dom_to_str(obs['dom_object']) # 
                    obs['text_content'] = self.html_text_converter.handle(html_str)
                    # make observation serializable
                    obs['set_of_marks'] = self.image_to_png_base64_url(
                        overlay_som(
                            obs['screenshot'], obs.get('extra_element_properties', {})
                        ),
                        add_data_prefix=True,
                    )
                    obs['screenshot'] = self.image_to_png_base64_url(
                        obs['screenshot'], add_data_prefix=True
                    )
                    obs['active_page_index'] = obs['active_page_index'].item()
                    obs['elapsed_time'] = obs['elapsed_time'].item()

                    # # Caching Logic: if it was a nav action and a cache miss, set the new observation in cache
                    # if url:
                    #     cache_client.set(url, obs)
                        
                    self.browser_side.send((unique_request_id, obs))
                    self.render_cache = obs
            except KeyboardInterrupt:
                logger.debug('Browser env process interrupted by user.')
                try:
                    env.close()
                except Exception:
                    pass
                return

    def step(self, action_str: str, timeout: float = 100) -> tuple[BrowserOutputObservation, float, bool, dict]:
        """
        Execute an action in the browser environment and return the observation.
        action_str: 可以是具体的 action 动作，也有可能是 外部传入的 最终答案， 会以这样的形式传入：<answer>{answer content}</answer>, 我们需要更具这个信息进行结果对比。
        """
        if action_str == "<invalid_action>":
            self.render_cache.update_error("Not a valid action.")
            return self.render_cache, 0, False, {}

        if "<answer>" in action_str:
            answer = action_str.split("<answer>")[1].split("</answer>")[0]
            is_correct = self.check_answer(answer, self.current_task.ground_truth)
            return None, 1 if is_correct else 0, True, {}

        unique_request_id = str(uuid.uuid4())
        self.agent_side.send((unique_request_id, {'action': action_str}))
        start_time = time.time()
        try:
            while True:
                if should_exit() or time.time() - start_time > timeout:
                    raise TimeoutError('Browser environment took too long to respond.')
                if self.agent_side.poll(timeout=0.01):
                    response_id, obs = self.agent_side.recv()
                    if response_id == unique_request_id: # TODO 限制了同步串行行为
                        observation = BrowserOutputObservation(**format_browser_observation(obs))
                        self.render_cache = observation
                        # TODO 具体每一轮次 action 的 reward 和 done 需要外部评估器评估，除了异常报错的问题
                        return observation, 0, False, {} # TODO 还有 reward, done, info 等额外信息需要添加。
        except Exception as e:
            logger.error(f'Encountered an error when executing browser action: {e}')
            observation = BrowserOutputObservation(
                content=str(e),
                screenshot='',
                error=True,
                last_browser_action_error=str(e),
                url='',
                trigger_by_action='browse_interactive',
            )
            self.render_cache = observation
            self.browser_retries += 1
            if self.browser_retries >= self.browser_retries_limit:
                return observation, 0, True, {}
            return observation, 0, False, {} # TODO 当出现异常报错的时候，是否可以放心评判，rewar= 0 和 done=Ture
        
    def check_answer(self, answer: str, ground_truth: str) -> bool:
        # TODO fake verifier 实现
        return True
                    

    def check_alive(self, timeout: float = 60):
        self.agent_side.send(('IS_ALIVE', None))
        if self.agent_side.poll(timeout=timeout):
            response_id, _ = self.agent_side.recv()
            if response_id == 'ALIVE':
                return True
            logger.debug(f'Browser env is not alive. Response ID: {response_id}')
        return False

    def close(self):
        if not self.process.is_alive():
            return
        try:
            self.agent_side.send(('SHUTDOWN', None))
            self.process.join(5)  # Wait for the process to terminate
            if self.process.is_alive():
                logger.error(
                    'Browser process did not terminate, forcefully terminating...'
                )
                self.process.terminate()
                self.process.join(5)  # Wait for the process to terminate
                if self.process.is_alive():
                    self.process.kill()
                    self.process.join(5)  # Wait for the process to terminate
            self.agent_side.close()
            self.browser_side.close()
        except Exception as e:
            logger.error(f'Encountered an error when closing browser env: {e}')

    @staticmethod
    def image_to_png_base64_url(
        image: np.ndarray | Image.Image, add_data_prefix: bool = False
    ):
        """Convert a numpy array to a base64 encoded png image url."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format='PNG')

        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        return (
            f'data:image/png;base64,{image_base64}'
            if add_data_prefix
            else f'{image_base64}'
        )

    @staticmethod
    def image_to_jpg_base64_url(
        image: np.ndarray | Image.Image, add_data_prefix: bool = False
    ):
        """Convert a numpy array to a base64 encoded jpeg image url."""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode in ('RGBA', 'LA'):
            image = image.convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format='JPEG')

        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        return (
            f'data:image/jpeg;base64,{image_base64}'
            if add_data_prefix
            else f'{image_base64}'
        )
    
    def render(self):
        return self.render_cache

    def reset(self, seed: Optional[int] = None, **kwargs: any) -> Any:
        with all_seed(seed):
            self.current_task_idx = random.randint(0, len(self.train_data) - 1)
        task = self.train_data['train'][self.current_task_idx]
        # print(f"current_task_idx: {self.current_task_idx} task: {task}")
        self.current_task = Task(
            task_idx=self.current_task_idx, 
            data_source=task["data_source"], 
            data_url=task["data_url"], 
            instruction=task["instruction"], 
            action_tip=task["action_tip"], 
            ground_truth=task["ground_truth"]
            )
        self.agent_side.send(('RESET', None)) # reset the browser to blank page
        start_time = time.time()
        reset_timeout = 120  # 设置超时时间，例如120秒
        while True:
            if should_exit() or time.time() - start_time > reset_timeout:
                logger.error(f"Timeout or exit signal received during reset after {reset_timeout} seconds.")
                raise TimeoutError('Browser environment took too long to respond during reset.')
            if self.agent_side.poll(timeout=0.01):
                response_id, obs = self.agent_side.recv()
                if response_id == 'RESET_DONE':
                    try:
                        self.render_cache = BrowserOutputObservation(**format_browser_observation(obs))
                        self.render_cache.add_task_goal(self.current_task.get_task_goal()) # reset env 后，第一次observation 添加 goal 信息
                        logger.info(f"Browser Reset done after {time.time() - start_time} seconds.")
                    except Exception as e:
                        logger.error(f"Error in WebBrowserEnv.reset when formatting browser observation: {e}")
                        self.render_cache = BrowserOutputObservation(
                            content=str(e),
                            screenshot='',
                            error=True,
                            last_browser_action_error=str(e),
                            url='',
                            trigger_by_action='browse_interactive',
                        )
                        self.render_cache.add_task_goal(self.current_task.get_task_goal()) # reset env 后，第一次observation 添加 goal 信息
                    break
            time.sleep(0.1) # 减少 sleep 时间以便更快响应
        
        return self.render()


if __name__ == '__main__':
    env = WebBrowserEnv(config=WebBrowserEnvConfig(high_level_action_set=["bid", "nav"]))
    obs = env.reset(seed=11)
    print('init obs: ', obs)
    print('init condensed obs: ', obs.get_condensed_observation())

    obs, reward, done, info = env.step(action_str="goto('https://www.stats.gov.cn/')")
    print(f"after goto obs: {obs}") # TODO 研究一下获取的 obs [16] link '', clickable, url='https://www.chinabgao.com/kf/dialog_1.htm?arg=9007904&style=1' 怎么控制 url 是否出现。
    print(f"after goto condensed obs: {obs.get_condensed_observation()}")
    print(f"render obs: {env.render()}")
    # with open('obs.txt', 'w') as f:
    #     f.write(str(obs))
    # print(reward)
    # print(done)
    # print(info)
    env.close()