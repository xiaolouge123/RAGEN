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
from pathlib import Path
import os
import json

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

# 自动登录配置
ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
    "shopping_admin": {"username": "admin", "password": "admin1234"},
}

# 从环境变量获取URL，如果没有设置则使用默认值
URLS = {
    "reddit": os.environ.get("REDDIT", "http://10.0.0.104:9999"),
    "shopping": os.environ.get("SHOPPING", "http://10.0.0.104:7770"),
    "shopping_admin": os.environ.get("SHOPPING_ADMIN", "http://10.0.0.104:7780/admin"),
    "gitlab": os.environ.get("GITLAB", "http://10.0.0.104:8023/explore/"),
}

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

def grep_action_items(action_str: str):
    # IMPORTANT: action_str 通过 \n 分割，所以需要处理 \n 的情况
    action_items = action_str.split("\n")
    valid_action, invalid_action = [], []
    for action_item in action_items:
        if len(action_item.strip()) == 0:
            continue
        action_name = action_item.strip().split("(")[0]
        if action_name in VALID_ACTIONS:
            valid_action.append(action_name)
        else:
            invalid_action.append(action_name)
    return valid_action, invalid_action


@dataclass
class Task:
    task_idx: int
    data_source: str
    data_url: str
    instruction: str
    action_tip: str
    ground_truth: str
    target_url: str
            
    def get_task_goal(self):
        if self.action_tip:
            return f"在{self.data_url}网站中，找到{self.instruction}，网站中寻找到目标数据的经验tips总结如下: {self.action_tip}"
        else:
            return f"在{self.data_url}网站中，找到{self.instruction}"


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
        self.mode = config.mode if config is not None else "train"
        self.html_text_converter = self.get_html_text_converter()
        self.config = config if config is not None else WebBrowserEnvConfig()
        self.data = datasets.load_dataset("parquet", data_files=self.config.train_path) if self.mode == "train" else datasets.load_dataset("parquet", data_files=self.config.val_path)
        self.browser_retries_limit = getattr(self.config, "browser_retries_limit", 3)

        # 添加自动登录配置
        self.auth_folder = getattr(self.config, "auth_folder", "./.auth")
        self.enable_auto_login = getattr(self.config, "enable_auto_login", False)
        
        # 登录状态缓存
        self._storage_states = {}
        self._load_storage_states()

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
    
    @staticmethod
    def create_auth_file(site: str, auth_folder: str, accounts: dict, urls: dict) -> str:
        from playwright.sync_api import sync_playwright
        import time
        print(f"🔐 [认证生成] 开始为站点 {site} 生成认证状态...")
        if site not in accounts:
            print(f"❌ [认证生成] 错误：站点 {site} 的账户信息不存在")
            return ""
        if site not in urls:
            print(f"❌ [认证生成] 错误：站点 {site} 的URL信息不存在")
            return ""
        os.makedirs(auth_folder, exist_ok=True)
        storage_state_path = os.path.join(auth_folder, f"{site}_state.json")
        try:
            with sync_playwright() as p:
                print(f"🔐 [认证生成] 启动浏览器...")
                browser = p.chromium.launch(headless=True)
                context = browser.new_context()
                page = context.new_page()
                page.set_default_timeout(120000)
                page.set_default_navigation_timeout(120000)
                if site == "shopping":
                    username = accounts["shopping"]["username"]
                    password = accounts["shopping"]["password"]
                    print(f"🔐 [认证生成] 正在登录 shopping 站点...")
                    page.goto(f"{urls['shopping']}/customer/account/login/", wait_until="domcontentloaded")
                    try:
                        page.wait_for_selector('input[placeholder="Email"]', timeout=120000)
                        page.get_by_placeholder("Email").fill(username)
                        page.wait_for_selector('input[placeholder="Password"]', timeout=120000)
                        page.get_by_placeholder("Password").fill(password)
                        page.wait_for_selector('button[type="submit"]', timeout=120000)
                        page.get_by_role("button", name="Sign In").click()
                    except Exception as e:
                        print(f"❌ [认证生成] shopping登录失败: {e}")
                        print(f"页面内容预览: {page.content()[:500]}...")
                        browser.close()
                        return ""
                elif site == "shopping_admin":
                    username = accounts["shopping_admin"]["username"]
                    password = accounts["shopping_admin"]["password"]
                    print(f"🔐 [认证生成] 正在登录 shopping_admin 站点...")
                    page.goto(urls["shopping_admin"], wait_until="domcontentloaded")
                    try:
                        page.wait_for_selector('input[placeholder="user name"]', timeout=120000)
                        page.get_by_placeholder("user name").fill(username)
                        page.wait_for_selector('input[placeholder="password"]', timeout=120000)
                        page.get_by_placeholder("password").fill(password)
                        page.wait_for_selector('button[type="submit"]', timeout=120000)
                        page.get_by_role("button", name="Sign in").click()
                    except Exception as e:
                        print(f"❌ [认证生成] shopping_admin登录失败: {e}")
                        print(f"页面内容预览: {page.content()[:500]}...")
                        browser.close()
                        return ""
                elif site == "reddit":
                    username = accounts["reddit"]["username"]
                    password = accounts["reddit"]["password"]
                    print(f"🔐 [认证生成] 正在登录 reddit 站点...")
                    page.goto(f"{urls['reddit']}/login", wait_until="domcontentloaded")
                    try:
                        page.wait_for_selector('input[name="username"]', timeout=120000)
                        page.get_by_label("Username").fill(username)
                        page.wait_for_selector('input[name="password"]', timeout=120000)
                        page.get_by_label("Password").fill(password)
                        page.wait_for_selector('button[type="submit"]', timeout=120000)
                        page.get_by_role("button", name="Log in").click()
                    except Exception as e:
                        print(f"❌ [认证生成] reddit登录失败: {e}")
                        print(f"页面内容预览: {page.content()[:500]}...")
                        browser.close()
                        return ""
                elif site == "gitlab":
                    username = accounts["gitlab"]["username"]
                    password = accounts["gitlab"]["password"]
                    print(f"🔐 [认证生成] 正在登录 gitlab 站点...")
                    page.goto(f"{urls['gitlab']}/users/sign_in", wait_until="domcontentloaded")
                    try:
                        page.wait_for_selector('[data-testid="username-field"]', timeout=120000)
                        page.get_by_test_id("username-field").fill(username)
                        page.get_by_test_id("username-field").press("Tab")
                        page.wait_for_selector('[data-testid="password-field"]', timeout=120000)
                        page.get_by_test_id("password-field").fill(password)
                        page.wait_for_selector('[data-testid="sign-in-button"]', timeout=120000)
                        page.get_by_test_id("sign-in-button").click()
                    except Exception as e:
                        print(f"❌ [认证生成] gitlab登录失败: {e}")
                        print(f"页面内容预览: {page.content()[:500]}...")
                        browser.close()
                        return ""
                else:
                    print(f"❌ [认证生成] 错误：不支持的站点 {site}")
                    browser.close()
                    return ""
                time.sleep(5)  # 等待cookie写入
                print(f"🔐 [认证生成] 保存认证状态到 {storage_state_path}")
                context.storage_state(path=storage_state_path)
                browser.close()
                if os.path.exists(storage_state_path):
                    print(f"✅ [认证生成] 成功为站点 {site} 生成认证状态")
                    return storage_state_path
                else:
                    print(f"❌ [认证生成] 认证状态文件未生成")
                    return ""
        except Exception as e:
            print(f"❌ [认证生成] 生成认证状态时发生错误: {e}")
            return ""
    
    def _load_storage_states(self):
        """加载认证状态文件"""
        if not self.enable_auto_login:
            return
            
        auth_path = Path(self.auth_folder)
        if not auth_path.exists():
            logger.warning(f"认证文件夹不存在: {auth_path}")
            return
            
        print(f"🔐 [自动登录] 正在加载认证状态文件从: {auth_path}")
        
        # 加载各个站点的认证状态
        for auth_file in auth_path.glob("*_state.json"):
            try:
                site_name = auth_file.stem.replace("_state", "")
                with open(auth_file, 'r') as f:
                    storage_state = json.load(f)
                self._storage_states[site_name] = {
                    "path": str(auth_file),
                    "data": storage_state
                }
                print(f"  ✓ 已加载 {site_name} 的认证状态")
            except Exception as e:
                logger.error(f"加载认证状态文件失败 {auth_file}: {e}")
        
        print(f"🔐 [自动登录] 共加载了 {len(self._storage_states)} 个认证状态")

    def _get_storage_state_for_task(self, task: Dict[str, Any]) -> Optional[str]:
        """根据任务获取对应的认证状态文件"""
        if not self.enable_auto_login:
            return None
            
        # 检查任务是否需要登录
        if not task.get("require_login", False):
            return None
            
        # 检查任务中是否指定了storage_state
        task_storage_state = task.get("storage_state", "")
        if task_storage_state:
            # 从任务配置中获取存储状态路径
            auth_file = Path(self.auth_folder) / Path(task_storage_state).name
            if auth_file.exists():
                print(f"🔐 [自动登录] 使用任务指定的认证状态: {auth_file}")
                return str(auth_file)
            if not auth_file.exists():
                # 自动生成cookie - 从文件名推断站点名称
                site_name = Path(task_storage_state).stem.replace("_state", "")
                print(f"🔐 [自动登录] 未找到cookie，自动登录生成: {auth_file}")
                # 你需要准备accounts和urls字典
                storage_state_path = self.create_auth_file(site_name, self.auth_folder, ACCOUNTS, URLS)
                if os.path.exists(storage_state_path):
                    return storage_state_path
                else:
                    print(f"❌ [自动登录] 自动登录失败，未生成cookie文件")
                    return None
        
        # 根据站点名称匹配认证状态
        sites = task.get("sites", [])
        if sites:
            site = sites[0]  # 使用第一个站点
            
            # 尝试精确匹配
            if site in self._storage_states:
                auth_path = self._storage_states[site]["path"]
                print(f"🔐 [自动登录] 为站点 {site} 找到认证状态: {auth_path}")
                return auth_path
                
            # 尝试模糊匹配
            for stored_site, state_info in self._storage_states.items():
                if site in stored_site or stored_site in site:
                    auth_path = state_info["path"]
                    print(f"🔐 [自动登录] 为站点 {site} 找到模糊匹配的认证状态: {auth_path} (匹配: {stored_site})")
                    return auth_path
        
        logger.warning(f"未找到适合任务的认证状态，任务站点: {sites}")
        return None
    
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
        # 检查是否需要自动登录
        storage_state_path = getattr(self, '_current_storage_state', None)
        
        browser_kwargs = {
            'task_kwargs': {'start_url': 'about:blank', 'goal': 'PLACEHOLDER_GOAL'},
            'wait_for_user_message': False,
            'headless': True,
            'disable_env_checker': True,
            'tags_to_mark': 'all',
            'enable_context_cache': ENABLE_CONTEXT_CACHE,
            'context_cache_kwargs': {"redis_url": REDIS_URL, "ttl": TTL, "cacheable_resource_types": CACHE_RESOURCE_TYPES},
            'resource_filter_kwargs': RESOURCE_FILTER_KWARGS,
        }
        
        # 如果有认证状态，添加到浏览器配置中
        if storage_state_path and Path(storage_state_path).exists():
            print(f"🔐 [浏览器进程] 使用认证状态启动浏览器: {storage_state_path}")
            browser_kwargs['task_kwargs']['storage_state'] = storage_state_path
        
        env = gym.make('browsergym/openended', **browser_kwargs)
        obs, info = env.reset()
        self.render_cache = obs
        logger.info('Successfully called env.reset')
        
        if storage_state_path:
            print(f"🔐 [浏览器进程] 浏览器已使用认证状态启动完成")
        
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

    def calculate_reward(
        self,
        current_url: str,
        action_str: str,
        observation: BrowserOutputObservation
    ) -> float:
        """
        根据当前URL与目标URL的对比计算 reward 分数（0~10）。
        采用过程奖励机制，鼓励模型逐步接近目标网站。
        
        评分策略：
        - 错误/无效动作: 0分
        - 普通动作阶段: 0-8分 (基础分+进步奖励)
        - 答案输出阶段: 2-10分 (根据最终匹配度)
        """
        if not self.current_task or not self.current_task.target_url:
            return 0.0

        target_url = self.current_task.target_url.strip()
        current_url = current_url.strip()

        print(f"[REWARD] 目标URL: {target_url}")
        print(f"[REWARD] 当前URL: {current_url}")
        print(f"[REWARD] 动作: {action_str}")

        # 1) 错误或无效动作 - 0分
        if observation.error:
            print(f"[REWARD] 检测到错误，给予0分")
            return 0.0
        
        if action_str == "<invalid_action>":
            print(f"[REWARD] 无效动作，给予0分")
            return 0.0

        # 2) <answer> 阶段 - 最终评估 (2-10分)
        if "<answer>" in action_str:
            print(f"[REWARD] 答案输出阶段，进行最终评估")
            if current_url == target_url:
                print(f"[REWARD] 完美匹配！给予10分")
                return 10.0
            
            similarity = self._calculate_url_similarity(current_url, target_url)
            if similarity >= 8.0:
                reward = 9.0
                print(f"[REWARD] 高度匹配(相似度:{similarity:.1f})，给予{reward}分")
            elif similarity >= 6.0:
                reward = 7.0
                print(f"[REWARD] 中等匹配(相似度:{similarity:.1f})，给予{reward}分")
            elif similarity >= 3.0:
                reward = 5.0
                print(f"[REWARD] 低匹配(相似度:{similarity:.1f})，给予{reward}分")
            else:
                reward = 2.0
                print(f"[REWARD] 很低匹配(相似度:{similarity:.1f})，给予{reward}分")
            return reward

        # 3) 普通动作阶段 - 过程奖励 (0-8分)
        current_similarity = self._calculate_url_similarity(current_url, target_url)
        print(f"[REWARD] 当前URL相似度: {current_similarity:.2f}")
        
        # 获取上一步的相似度
        previous_similarity = getattr(self, '_previous_url_similarity', 0.0)
        print(f"[REWARD] 上一步URL相似度: {previous_similarity:.2f}")
        
        # 基础分数：基于当前相似度 (0-5分)
        base_score = (current_similarity / 10.0) * 5.0
        
        # 进步奖励：如果比上一步更接近目标 (0-3分)
        progress = current_similarity - previous_similarity
        if progress > 0:
            progress_bonus = min(progress * 0.5, 3.0)  # 进步奖励最多3分
            print(f"[REWARD] 检测到进步: +{progress:.2f}，进步奖励: {progress_bonus:.2f}")
        else:
            progress_bonus = 0.0
            if progress < 0:
                print(f"[REWARD] 相似度下降: {progress:.2f}")
            else:
                print(f"[REWARD] 相似度无变化")
        
        # 更新历史记录
        self._previous_url_similarity = current_similarity
        
        # 总分计算
        total_reward = base_score + progress_bonus
        total_reward = min(total_reward, 8.0)  # 普通阶段最高8分，为answer阶段留出空间
        
        print(f"[REWARD] 基础分: {base_score:.2f}, 进步奖励: {progress_bonus:.2f}, 总分: {total_reward:.2f}")
        return total_reward

    def _calculate_url_similarity(self, current_url: str, target_url: str) -> float:
        """
        计算两个URL的相似度，返回0-10分
        
        评分维度：
        - 域名匹配: 0-6分
        - 路径匹配: 0-4分
        """
        if not current_url or current_url in ("about:blank", ""):
            return 0.0
        
        if current_url == target_url:
            return 10.0
        
        from urllib.parse import urlparse
        
        try:
            curr_parsed = urlparse(current_url)
            target_parsed = urlparse(target_url)
            
            # 域名评分 (0-6分)
            domain_score = self._calculate_domain_similarity(curr_parsed.netloc, target_parsed.netloc)
            
            # 路径评分 (0-4分)
            path_score = self._calculate_path_similarity(curr_parsed.path, target_parsed.path)
            
            total_similarity = domain_score + path_score
            return min(total_similarity, 10.0)
            
        except Exception as e:
            print(f"[REWARD] URL解析失败: {e}")
            # 降级处理：简单字符串匹配
            if target_url in current_url or current_url in target_url:
                return 3.0
            return 0.0

    def _calculate_domain_similarity(self, current_domain: str, target_domain: str) -> float:
        """
        计算域名相似度 (0-6分)
        
        评分规则：
        - 完全匹配: 6分
        - 同主域名不同子域: 4分  
        - 包含关系: 2分
        - 无关系: 0分
        """
        if not current_domain or not target_domain:
            return 0.0
        
        if current_domain == target_domain:
            return 6.0
        
        # 分割域名部分
        curr_parts = current_domain.split('.')
        target_parts = target_domain.split('.')
        
        # 检查主域名是否相同 (example.com)
        if len(curr_parts) >= 2 and len(target_parts) >= 2:
            curr_main = '.'.join(curr_parts[-2:])
            target_main = '.'.join(target_parts[-2:])
            
            if curr_main == target_main:
                return 4.0  # 同主域名，不同子域名
        
        # 检查包含关系
        if target_domain in current_domain or current_domain in target_domain:
            return 2.0
        
        return 0.0

    def _calculate_path_similarity(self, current_path: str, target_path: str) -> float:
        """
        计算路径相似度 (0-4分)
        
        评分规则：
        - 路径完全匹配: 4分
        - 都是根路径: 4分
        - 基于公共前缀比例: 0-4分
        - 包含关系: 至少2分
        """
        current_path = current_path.strip('/')
        target_path = target_path.strip('/')
        
        if current_path == target_path:
            return 4.0
        
        if not target_path and not current_path:
            return 4.0  # 都是根路径
        
        if not target_path:
            return 2.0  # 目标是根路径，当前有路径
        
        if not current_path:
            return 1.0  # 当前是根路径，目标有路径
        
        # 分割路径段
        curr_parts = [p for p in current_path.split('/') if p]
        target_parts = [p for p in target_path.split('/') if p]
        
        if not curr_parts or not target_parts:
            return 1.0
        
        # 计算公共前缀
        common_parts = 0
        for c, t in zip(curr_parts, target_parts):
            if c == t:
                common_parts += 1
            else:
                break
        
        # 基于公共前缀比例计算相似度
        similarity_ratio = common_parts / len(target_parts)
        path_score = similarity_ratio * 4.0
        
        # 检查包含关系
        if target_path in current_path or current_path in target_path:
            path_score = max(path_score, 2.0)
        
        return path_score


    # def calculate_reward(
    #     self,
    #     current_url: str,
    #     action_str: str,
    #     observation: BrowserOutputObservation,
    # ) -> float:
    #     """
    #     0–10 分直接给分：
    #     • 错误或无效动作立即返回 0
    #     • 普通阶段：根据 URL 相似度给“过程奖励”= max(sim_cur - sim_prev, 0)
    #     • <answer> 阶段：若真正到达目标页，再给一次 10 分满分
    #     """
    #     # ---------- 安全检查 ----------
    #     if (not self.current_task or
    #         not self.current_task.target_url or
    #         not current_url):
    #         return 0.0

    #     target_url  = self.current_task.target_url.strip()
    #     current_url = current_url.strip()

    #     # ---------- ① 处理错误 / 无效动作 ----------
    #     if observation.error or action_str == "<invalid_action>":
    #         return 0.0

    #     # ---------- ② 计算 URL“接近度” 0–10 ----------
    # #     sim_cur = self._url_similarity(current_url, target_url)   # 0–10
    #     sim_cur = self._url_similarity(current_url, target_url)   # 0–10

    #     # ---------- ③ 如果进入 <answer> 阶段 ----------
    #     if "<answer>" in action_str:
    #         # 只有真正到达目标，才给满分；否则给 3 分象征鼓励
    #         final_reward = 10.0 if current_url == target_url else 3.0
    #         self._answer_reached = True
    #         self._prev_similarity = sim_cur     # 更新状态，防止后续误差
    #         return final_reward

    # #     # ---------- ④ 普通阶段：过程奖励 ----------
    # #     #   只奖励“越来越接近”的正向增量
    # #     reward = max(sim_cur - getattr(self, "_prev_similarity", 0.0), 0.0)
    # #     self._prev_similarity = sim_cur
    # #     return reward
    #         # ---------- ④ 普通阶段：过程奖励 ----------
    #     # ① 基础分：sim_cur 的 10%（确保即便“停在高相似度”也有分）
    #     # ② 增量分：正向增量 × 90%
    #     prev = getattr(self, "_prev_similarity", 0.0)
    #     base      = 0.1 * sim_cur
    #     increment = 0.9 * max(sim_cur - prev, 0.0)
    #     reward = base + increment

    #     # 更新状态
    #     self._prev_similarity = sim_cur
    #     return reward


    # # # ======== 辅助函数 ========
    # # def _url_similarity(self, cur: str, tgt: str) -> float:
    # #     """
    # #     把 URL 拆成域名 + 路径两部分，各自判定：
    # #     • 域名：完全相同 6 分 / 子域包含 3 分 / 不同 0 分
    # #     • 路径：按公共前缀段数占比 × 4 分
    # #     最终结果 ∈ [0, 10]
    # #     """
    # #     cur_p = urlparse(cur)
    # #     tgt_p = urlparse(tgt)

    # #     # ---------- 域名部分 ----------
    # #     if cur_p.netloc == tgt_p.netloc:
    # #         domain_score = 6.0
    # #     elif cur_p.netloc.endswith(tgt_p.netloc) or tgt_p.netloc.endswith(cur_p.netloc):
    # #         domain_score = 3.0          # 同一主域 + 不同子域
    # def _url_similarity(self, cur, tgt):
    #     cur_p, tgt_p = urlparse(cur), urlparse(tgt)
    #     cur_dom, tgt_dom = cur_p.netloc, tgt_p.netloc

    #     # ① 顶级域相同（比如 example.com vs sub.example.com）
    #     if cur_dom == tgt_dom:
    #         domain_score = 6.0
    #     elif cur_dom.split('.')[-2:] == tgt_dom.split('.')[-2:]:
    #         domain_score = 2.0    # 同一顶级域，放宽一点
    #     else:
    #         domain_score = 0.0

    #     # ---------- 路径部分 ----------
    #     cur_parts = [p for p in cur_p.path.split('/') if p]
    #     tgt_parts = [p for p in tgt_p.path.split('/') if p]
    #     if not tgt_parts:                        # 目标无路径 → 省略比较
    #         path_score = 4.0 if not cur_parts else 2.0
    #     else:
    #         common = 0
    #         for c, t in zip(cur_parts, tgt_parts):
    #             if c == t:
    #                 common += 1
    #             else:
    #                 break
    #         ratio = common / len(tgt_parts)      # 0–1
    #         path_score = ratio * 4.0             # 0–4

    #     similarity = domain_score + path_score
    #     return similarity                        # 已天然处于 0–10
        

    def step(self, action_str: str, timeout: float = 30) -> tuple[BrowserOutputObservation, float, bool, dict]:  # 从100秒减少到30秒
        """
        Execute an action in the browser environment and return the observation.
        action_str: 可以是具体的 action 动作，也有可能是 外部传入的 最终答案， 会以这样的形式传入：<answer>{answer content}</answer>, 我们需要更具这个信息进行结果对比。
        """
        if action_str == "<invalid_action>":
            self.render_cache.update_error("Not a valid action. Stay in the same page.")
            reward = self.calculate_reward(self.render_cache.url, action_str, self.render_cache)
            print(f"🎮 [STEP结果] 无效动作处理完成，最终reward: {reward}")
            return self.render_cache, reward, False, {"meta_info": {"status": "action_error", "msg": "Not a valid action."}}

        if "<answer>" in action_str:
            answer = action_str.split("<answer>")[1].split("</answer>")[0]
            observation = BrowserOutputObservation(
                content="Thank you for your answer.",
                screenshot='',
                error=False,
                last_browser_action_error="",
                url='',
                trigger_by_action='browse_interactive',
                final_answer=answer,
            )
            self.render_cache = observation
            reward = self.calculate_reward(self.render_cache.url, action_str, observation)
            print(f"🎮 [STEP结果] 答案输出处理完成，最终reward: {reward}, 任务结束!")
            return observation, reward, True, {"meta_info": {"status": "answer_output", "msg": "answer turn."}} # 宣告任务结束
        
        time.sleep(random.uniform(0.1, 0.5)) # 大幅减少sleep时间：从1-3秒改为0.1-0.5秒
        unique_request_id = str(uuid.uuid4())
        valid_action, invalid_action = grep_action_items(action_str)
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
                        # 计算基于URL对比的reward
                        reward = self.calculate_reward(observation.url, action_str, observation)
                        print(f"🎮 [STEP结果] 普通动作执行完成，最终reward: {reward}")
                        return observation, reward, False, {"meta_info": {"status": "action_output", "msg": "action turn", "valid_action": valid_action, "invalid_action": invalid_action}}
        except Exception as e:
            logger.error(f'Encountered an error when executing browser action: {e}, input action: {action_str}')
            observation = BrowserOutputObservation(
                content=str(e),
                screenshot='',
                error=True,
                last_browser_action_error=str(e),
                url='',
                trigger_by_action='browse_interactive',
            )
            self.render_cache = observation
            reward = self.calculate_reward(observation.url, action_str, observation)
            print(f"🎮 [STEP结果] 异常处理完成，最终reward: {reward}")
            return observation, reward, False, {"meta_info": {"status": "exception", "msg": "action turn", "valid_action": valid_action, "invalid_action": invalid_action}}
                    

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
        # 持续更新当前 observation
        return self.render_cache

    def reset(self, seed: Optional[int] = None, **kwargs: any) -> Any:
        global_step = kwargs.get("global_step", 0)
        total_steps = kwargs.get("total_steps", 0)
        progress = global_step / total_steps if total_steps > 0 else 0
        inv_progress = min(1 - progress, 0.1)
        dummy_task = random.random() < inv_progress
        
        with all_seed(seed):
            self.current_task_idx = random.randint(0, len(self.data['train']) - 1)
        task = self.data['train'][self.current_task_idx]
        
        # 检查是否需要自动登录
        storage_state_path = self._get_storage_state_for_task(task)
        if storage_state_path:
            print(f"🔐 [重置] 任务需要自动登录，设置认证状态: {storage_state_path}")
            self._current_storage_state = storage_state_path
        else:
            self._current_storage_state = None
            if task.get("require_login", False):
                print(f"⚠️ [重置] 任务需要登录但未找到认证状态文件")
        
        if dummy_task:
            logger.info(f"Easy mode, task start at target url: {task['target_url']}")
        logger.info(f"Resetting browser env with seed: {seed}, task_id: {self.current_task_idx}")
        
        self.current_task = Task(
            task_idx=self.current_task_idx, 
            data_source=task["data_source"], 
            data_url=task["data_url"] if not dummy_task else task["target_url"], 
            instruction=task["instruction"], 
            action_tip=task["action_tip"], 
            ground_truth=task["ground_truth"],
            target_url=task["target_url"],
            )
        
        # 初始化过程奖励跟踪变量
        self._previous_url_similarity = 0.0
        
        self.agent_side.send(('RESET', None)) # reset the browser to blank page
        start_time = time.time()
        reset_timeout = 60  # 从120秒减少到60秒
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
                        
                        # 如果需要自动登录，可以在这里执行一些额外的登录验证
                        if self._current_storage_state:
                            print(f"🔐 [重置完成] 浏览器已使用认证状态重置完成")
                        
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
            time.sleep(0.05) # 从0.1秒减少到0.05秒以便更快响应
        
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