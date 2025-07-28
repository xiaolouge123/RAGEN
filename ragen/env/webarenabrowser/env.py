import os
import time
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Any

# register openended gym environments
import browsergym.core
from browsergym.core import _get_global_playwright
import gymnasium as gym
from browsergym.utils.obs import flatten_dom_to_str, overlay_som

from ragen.log import openhands_logger as logger
from ragen.env.webbrowser.utils import stop_if_should_exit, should_continue, should_exit
from ragen.env.webbrowser.exception import BrowserInitException
from ragen.env.base import BaseLanguageBasedEnv
from ragen.env.webbrowser.env import WebBrowserEnv, format_browser_observation
from ragen.env.webarenabrowser.config import (
    WebArenaBrowserEnvConfig,
    ACCOUNTS,
    GITLAB,
    SHOPPING,
    SHOPPING_ADMIN,
    REDDIT,
    WIKIPEDIA
)   
from ragen.utils import all_seed
from ragen.llm_agent.observations import BrowserOutputObservation



ENABLE_CONTEXT_CACHE = True
REDIS_URL = "redis://localhost:6380/1"
TTL = 3600 # 1 hour
CACHE_RESOURCE_TYPES = ["document", "stylesheet", "script", "image", "font", "xhr", "fetch"]
RESOURCE_FILTER_KWARGS = {
                    'resource_filter_config': {
                        'block_images': True,
                        'allow_essential_images': True,
                        'block_videos': True,           # 阻止视频
                        'block_ads': True,              # 阻止广告
                        'block_analytics': True,        # 阻止分析脚本
                        'block_fonts': False,           # 保留字体以免影响显示
                        'custom_url_patterns': [        # 自定义阻止模式
                            r'.*\.gif$',                # 阻止GIF动图
                            r'.*banner.*',              # 阻止横幅
                            r'.*tracking.*',            # 阻止跟踪
                        ]
                    }
                }

VALID_ACTIONS = ["goto", "go_back", "go_forward", "noop", "scroll", "fill", "select_option", "click", "dblclick", "hover", "press", "focus", "clear", "drag_and_drop", "upload_file"]

SITES = ["gitlab", "shopping", "shopping_admin", "reddit"]
URLS = [
    f"{GITLAB}/-/profile",
    f"{SHOPPING}/wishlist/",
    f"{SHOPPING_ADMIN}/dashboard",
    f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
]
EXACT_MATCH = [True, True, True, True]
KEYWORDS = ["", "", "Dashboard", "Delete"]


@dataclass
class Task:
    task_idx: int
    data_source: str
    data_url: str
    instruction: str
    action_tip: str
    ground_truth: str
    target_url: str
    require_login: bool
    storage_state: str
            
    def get_task_goal(self):
        return f"Go to {self.data_url} and then {self.instruction}"
    

# class DummyWrapper(gym.Wrapper):
#     def reset(self, *, seed=None, options=None, **kwargs):
#         print("DummyWrapper got kwargs:", kwargs)
#         return self.env.reset(seed=seed, options=options, **kwargs)
    
def monkey_patch_new_reset_with_kwargs(self, *, seed=None, options=None, **kwargs):
    """A new reset function that accepts and forwards kwargs."""
    self._has_reset = True
    return self.env.reset(seed=seed, options=options, **kwargs)


class WebArenaBrowserEnv(WebBrowserEnv):
    def __init__(self, config: Optional[WebArenaBrowserEnvConfig] = None, **kwargs: any) -> None:
        super().__init__(config, **kwargs)

    def is_expired(
        self,
        storage_state: Path, 
        url: str, 
        keyword: str, 
        url_exact: bool = True
    ) -> bool:
        """Test whether the cookie is expired"""
        if not storage_state.exists():
            return True # expired

        pw = _get_global_playwright()
        browser = pw.chromium.launch(headless=True, slow_mo=0)
        context = browser.new_context(storage_state=storage_state)
        page = context.new_page()
        page.goto(url)
        time.sleep(1)
        d_url = page.url
        content = page.content()
        if keyword:
            return keyword not in content
        else:
            if url_exact:
                return d_url != url
            else:
                return url not in d_url
            
    def login(self,comb: list[str], storage_state_path: str) -> None:
        # current pretend only single website login

        pw = _get_global_playwright()
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        if "shopping" in comb:
            username = ACCOUNTS["shopping"]["username"]
            password = ACCOUNTS["shopping"]["password"]
            page.goto(f"{SHOPPING}/customer/account/login/")
            page.get_by_label("Email", exact=True).fill(username)
            page.get_by_label("Password", exact=True).fill(password)
            page.get_by_role("button", name="Sign In").click()

        if "reddit" in comb:
            username = ACCOUNTS["reddit"]["username"]
            password = ACCOUNTS["reddit"]["password"]
            page.goto(f"{REDDIT}/login")
            page.get_by_label("Username").fill(username)
            page.get_by_label("Password").fill(password)
            page.get_by_role("button", name="Log in").click()

        if "shopping_admin" in comb:
            username = ACCOUNTS["shopping_admin"]["username"]
            password = ACCOUNTS["shopping_admin"]["password"]
            page.goto(f"{SHOPPING_ADMIN}")
            page.get_by_placeholder("user name").fill(username)
            page.get_by_placeholder("password").fill(password)
            page.get_by_role("button", name="Sign in").click()

        if "gitlab" in comb:
            username = ACCOUNTS["gitlab"]["username"]
            password = ACCOUNTS["gitlab"]["password"]
            page.goto(f"{GITLAB}/users/sign_in")
            page.get_by_test_id("username-field").click()
            page.get_by_test_id("username-field").fill(username)
            page.get_by_test_id("username-field").press("Tab")
            page.get_by_test_id("password-field").fill(password)
            page.get_by_test_id("sign-in-button").click()

        context.storage_state(path=storage_state_path)


    def browser_process(self):
        gym.wrappers.OrderEnforcing.reset = monkey_patch_new_reset_with_kwargs
        env = gym.make(
            'browsergym/openended',
            task_kwargs={'start_url': 'about:blank', 'goal': 'PLACEHOLDER_GOAL'}, # 永远入口页面都是空白页，reset 后也是空白页。
            wait_for_user_message=False,
            headless=True,
            disable_env_checker=True,
            tags_to_mark='all', # TODO playwright context 缓存是输入参数记得实例化
            enable_context_cache=ENABLE_CONTEXT_CACHE,
            context_cache_kwargs={"redis_url": REDIS_URL, "ttl": TTL, "cacheable_resource_types": CACHE_RESOURCE_TYPES},
            resource_filter_kwargs=RESOURCE_FILTER_KWARGS,
        )
        # env = DummyWrapper(env)
        obs, info = env.reset() # 这个环境在 browsergym.core.env 中定义 BrowserEnv.reset
        self.render_cache = obs
        logger.info('Successfully called env.reset in browser_process')
        
        logger.info('Browser env started in browser_process.')

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
                        if action_data:
                            pw_context_kwargs = action_data
                            logger.info(f"Reset with context kwargs: {pw_context_kwargs}")
                            storage_state = pw_context_kwargs.get('storage_state', None)
                            if storage_state:
                                obs, info = env.reset(storage_state=storage_state)
                            else:
                                obs, info = env.reset()
                        else:
                            obs, info = env.reset()
                        
                        # TODO 考虑封装一下
                        html_str = flatten_dom_to_str(obs['dom_object'])
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
                    elif unique_request_id == 'LOGIN':
                        logger.info('Login recv, try to login...')
                        cur_site, storage_state_path = action_data
                        login_succ = self.login(cur_site, storage_state_path)
                        self.browser_side.send(('LOGIN', login_succ))
                        continue
                    elif unique_request_id == 'IS_EXPIRED':
                        logger.info('IS_EXPIRED recv, try to check if expired...')
                        storage_state, url, keyword, match = action_data
                        is_expired = self.is_expired(storage_state, url, keyword, match)
                        self.browser_side.send(('IS_EXPIRED', is_expired))
                        continue

                    action = action_data['action']
                    obs, reward, terminated, truncated, info = env.step(action)
                    # obs 包含如下字段： chat_messages,goal,goal_object, open_pages_urls,open_pages_titles,active_page_index,url,screenshot,dom_object,axtree_object,extra_element_properties,focused_element_bid,last_action,last_action_error,elapsed_time

                    # TODO 考虑封装一下
                    # add text content of the page
                    html_str = flatten_dom_to_str(obs['dom_object'])
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

    def reset(self, seed: Optional[int] = None, **kwargs: any) -> Any:
        with all_seed(seed):
            self.current_task_idx = random.randint(0, len(self.data['train']) - 1)
        task = self.data['train'][self.current_task_idx]
        
        if task['data_source'] == 'shopping' and "__SHOPPING__" in task['data_url']:
            data_url = task['data_url'].replace("__SHOPPING__", SHOPPING)
        elif task['data_source'] == 'shopping_admin' and "__SHOPPING_ADMIN__" in task['data_url']:
            data_url = task['data_url'].replace("__SHOPPING_ADMIN__", SHOPPING_ADMIN)
        elif task['data_source'] == 'gitlab' and "__GITLAB__" in task['data_url']:
            data_url = task['data_url'].replace("__GITLAB__", GITLAB)
        elif task['data_source'] == 'reddit' and "__REDDIT__" in task['data_url']:
            data_url = task['data_url'].replace("__REDDIT__", REDDIT)
        elif task['data_source'] == 'wikipedia' and "__WIKIPEDIA__" in task['data_url']:
            data_url = task['data_url'].replace("__WIKIPEDIA__", WIKIPEDIA)
        else:
            data_url = task['data_url']
        
        self.current_task = Task(
            task_idx=self.current_task_idx, 
            data_source=task["data_source"], 
            data_url=data_url, 
            instruction=task["instruction"], 
            action_tip=task["action_tip"], 
            ground_truth=task["ground_truth"],
            target_url=task["target_url"],
            require_login=task["require_login"],
            storage_state=task["storage_state"],
            )
        
        # 在reset 前，解决好 context 问题
        pw_context_kwargs = None
        cur_site = self.current_task.data_source
        url = URLS[SITES.index(cur_site)]
        keyword = KEYWORDS[SITES.index(cur_site)]
        match = EXACT_MATCH[SITES.index(cur_site)]
        storage_state_path = Path(self.config.storage_state_dir) / f"{cur_site}_state.json"
        
        if self.current_task.require_login:
            is_expired = False
            login_time_out = 60
            start_time = time.time()
            self.agent_side.send(('IS_EXPIRED', (storage_state_path, url, keyword, match)))
            while True:
                if should_exit() or time.time() - start_time > login_time_out:
                    logger.error(f"Timeout or exit signal received during IS_EXPIRED check after {login_time_out} seconds.")
                    raise TimeoutError('Browser environment took too long to respond during IS_EXPIRED check.')
                if self.agent_side.poll(timeout=0.01):
                    unique_request_id, obs = self.agent_side.recv()
                    if unique_request_id == 'IS_EXPIRED':
                        is_expired = obs
                        break
            logger.info(f"IS_EXPIRED result: {is_expired}")
            
            if is_expired:
                _start_time = time.time()
                self.agent_side.send(('LOGIN', ([cur_site], storage_state_path)))
                while True:
                    if should_exit() or time.time() - _start_time > login_time_out:
                        logger.error(f"Timeout or exit signal received during LOGIN after {login_time_out} seconds.")
                        raise TimeoutError('Browser environment took too long to respond during LOGIN.')
                    if self.agent_side.poll(timeout=0.01):
                        unique_request_id, obs = self.agent_side.recv()
                        if unique_request_id == 'LOGIN':
                            login_succ = obs
                            if login_succ:
                                logger.info("Login success")
                                break
                            else:
                                logger.error("Login failed")
                                raise Exception("Login failed")
        if os.path.exists(storage_state_path):
            pw_context_kwargs = {"storage_state": str(storage_state_path)}
        
        # origin reset logic
        self.agent_side.send(('RESET', pw_context_kwargs)) # reset the browser to blank page
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
                        self.render_cache.add_task({
                            'goal': self.current_task.get_task_goal(),
                            'ground_truth': self.current_task.ground_truth,
                        }) # reset env 后，第一次observation 添加 goal, ground_truth 等相关信息
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
                        self.render_cache.add_task({
                            'goal': self.current_task.get_task_goal(),
                            'ground_truth': self.current_task.ground_truth,
                        }) # reset env 后，第一次observation 添加 goal, ground_truth 等相关信息
                    break
            time.sleep(0.1) # 减少 sleep 时间以便更快响应
        
        return self.render()
        