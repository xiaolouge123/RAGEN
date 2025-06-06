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
from typing import Optional, Any
from dataclasses import dataclass, field


# register openended gym environments
import browsergym.core
import gymnasium as gym
from browsergym.utils.obs import flatten_dom_to_str, overlay_som, flatten_axtree_to_str

from ragen.log import openhands_logger as logger
from ragen.env.webbrowser.utils import stop_if_should_exit, should_continue, should_exit
from ragen.env.webbrowser.exception import BrowserInitException
from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.webbrowser.config import WebBrowserEnvConfig
from ragen.utils import all_seed


@dataclass
class BrowserOutputObservation:
    """This data class represents the output of a browser."""
    content: str
    url: str
    trigger_by_action: str
    screenshot: str = field(repr=False, default='')  # don't show in repr
    set_of_marks: str = field(default='', repr=False)  # don't show in repr
    error: bool = False
    observation: str = 'browse'
    goal_image_urls: list = field(default_factory=list)
    # do not include in the memory
    open_pages_urls: list = field(default_factory=list)
    active_page_index: int = -1
    dom_object: dict = field(default_factory=dict, repr=False)  # don't show in repr
    axtree_object: dict = field(default_factory=dict, repr=False)  # don't show in repr
    extra_element_properties: dict = field(
        default_factory=dict, repr=False
    )  # don't show in repr
    last_browser_action: str = ''
    last_browser_action_error: str = ''
    focused_element_bid: str = ''

    @property
    def message(self) -> str:
        return 'Visited ' + self.url

    def __str__(self) -> str:
        ret = (
            '**BrowserOutputObservation**\n'
            f'URL: {self.url}\n'
            f'Error: {self.error}\n'
            f'Open pages: {self.open_pages_urls}\n'
            f'Active page index: {self.active_page_index}\n'
            f'Last browser action: {self.last_browser_action}\n'
            f'Last browser action error: {self.last_browser_action_error}\n'
            f'Focused element bid: {self.focused_element_bid}\n'
        )
        ret += '--- Agent Observation ---\n'
        ret += self.get_agent_obs_text()
        return ret

    def get_agent_obs_text(self) -> str:
        """Get a concise text that will be shown to the agent."""
        if self.trigger_by_action == 'browse_interactive':
            text = f'[Current URL: {self.url}]\n'
            text += f'[Focused element bid: {self.focused_element_bid}]\n\n'
            if self.error:
                text += (
                    '================ BEGIN error message ===============\n'
                    'The following error occurred when executing the last action:\n'
                    f'{self.last_browser_action_error}\n'
                    '================ END error message ===============\n'
                )
            else:
                text += '[Action executed successfully.]\n'
            try:
                # We do not filter visible only here because we want to show the full content
                # of the web page to the agent for simplicity.
                # FIXME: handle the case when the web page is too large
                cur_axtree_txt = self.get_axtree_str(filter_visible_only=False)
                text += (
                    f'============== BEGIN accessibility tree ==============\n'
                    f'{cur_axtree_txt}\n'
                    f'============== END accessibility tree ==============\n'
                )
            except Exception as e:
                text += (
                    f'\n[Error encountered when processing the accessibility tree: {e}]'
                )
            return text

        elif self.trigger_by_action == 'browse':
            text = f'[Current URL: {self.url}]\n'
            if self.error:
                text += (
                    '================ BEGIN error message ===============\n'
                    'The following error occurred when trying to visit the URL:\n'
                    f'{self.last_browser_action_error}\n'
                    '================ END error message ===============\n'
                )
            text += '============== BEGIN webpage content ==============\n'
            text += self.content
            text += '\n============== END webpage content ==============\n'
            return text
        else:
            raise ValueError(f'Invalid trigger_by_action: {self.trigger_by_action}')

    def get_axtree_str(self, filter_visible_only: bool = False) -> str:
        cur_axtree_txt = flatten_axtree_to_str(
            self.axtree_object,
            extra_properties=self.extra_element_properties,
            with_clickable=True,
            skip_generic=False,
            filter_visible_only=filter_visible_only,
        )
        return cur_axtree_txt


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

        # Initialize browser environment process
        multiprocessing.set_start_method('spawn', force=True)
        self.browser_side, self.agent_side = multiprocessing.Pipe()

        self.init_browser()
        atexit.register(self.close)

        self.render_cache: dict = None
        self.current_task: str = None

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
                        obs, info = env.reset() # TODO 考虑下这里的入参seed要考虑一下怎么操作吗
                        self.render_cache = obs
                        self.browser_side.send(('RESET_DONE', None))
                        continue

                    action = action_data['action']
                    obs, reward, terminated, truncated, info = env.step(action)
                    # obs 包含如下字段： chat_messages,goal,goal_object, open_pages_urls,open_pages_titles,active_page_index,url,screenshot,dom_object,axtree_object,extra_element_properties,focused_element_bid,last_action,last_action_error,elapsed_time


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
                    self.browser_side.send((unique_request_id, obs))
                    self.render_cache = obs
            except KeyboardInterrupt:
                logger.debug('Browser env process interrupted by user.')
                try:
                    env.close()
                except Exception:
                    pass
                return

    def step(self, action_str: str, timeout: float = 100) -> dict:
        """Execute an action in the browser environment and return the observation."""
        unique_request_id = str(uuid.uuid4())
        self.agent_side.send((unique_request_id, {'action': action_str}))
        start_time = time.time()
        try:
            while True:
                if should_exit() or time.time() - start_time > timeout:
                    raise TimeoutError('Browser environment took too long to respond.')
                if self.agent_side.poll(timeout=0.01):
                    response_id, obs = self.agent_side.recv()
                    if response_id == unique_request_id:
                        observation = BrowserOutputObservation(**format_browser_observation(obs))
                        return str(observation), None, None, None # TODO 还有 reward,done,info 等额外信息需要添加。
        except Exception as e:
            logger.error(f'Encountered an error when executing browser action: {e}')
            observation = BrowserOutputObservation(
                content=str(e),
                screenshot='',
                error=True,
                last_browser_action_error=str(e),
                url='',
                trigger_by_action=action_str,
            )
            return str(observation), None, None, None # TODO 还有 reward,done,info 等额外信息需要添加。
                    

    def check_alive(self, timeout: float = 60):
        self.agent_side.send(('IS_ALIVE', None))
        if self.agent_side.poll(timeout=timeout):
            response_id, _ = self.agent_side.recv()
            if response_id == 'ALIVE':
                return True
            logger.debug(f'Browser env is not alive. Response ID: {response_id}')

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
    
    def _get_task(self, raw_task: dict):
        # data feilds: 'data_source', 'data_url', 'instruction', 'action_tip', 'ground_truth'
        return f"please find the data of {raw_task['instruction']} from {raw_task['data_url']}, given the tip: {raw_task['action_tip']}"
    
    def _get_ground_truth(self, raw_task: dict):
        return raw_task['ground_truth']

    def reset(self, seed: Optional[int] = None, **kwargs: any) -> Any:
        with all_seed(seed):
            task_idx = random.randint(0, len(self.train_data) - 1)
            self.current_task = self._get_task(self.train_data['train'][task_idx])
            self.current_ground_truth = self._get_ground_truth( self.train_data['train'][task_idx])
        
        self.agent_side.send(('RESET', None))
        start_time = time.time()
        reset_timeout = 30  # 设置超时时间，例如30秒
        while True:
            if should_exit() or time.time() - start_time > reset_timeout:
                logger.error(f"Timeout or exit signal received during reset after {reset_timeout} seconds.")
                raise TimeoutError('Browser environment took too long to respond during reset.')
            if self.agent_side.poll(timeout=0.01):
                response_id, obs = self.agent_side.recv()
                if response_id == 'RESET_DONE':
                    logger.info(f"Browser Reset done after {time.time() - start_time} seconds.")
                    break
            time.sleep(0.1) # 减少 sleep 时间以便更快响应
        
        return self.render()


if __name__ == '__main__':
    env = WebBrowserEnv(config=WebBrowserEnvConfig(high_level_action_set=["bid", "nav"]))
    print(env.reset())
    obs, reward, done, info = env.step(action_str="goto('https://m.chinabgao.com/stat/')")
    print(obs) # TODO 研究一下获取的 obs [16] link '', clickable, url='https://www.chinabgao.com/kf/dialog_1.htm?arg=9007904&style=1' 怎么控制 url 是否出现。
    print(reward)
    print(done)
    print(info)
    env.close()