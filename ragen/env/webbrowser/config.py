from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List  

@dataclass
class WebBrowserEnvConfig:
    """Configuration for WebBrowserEnv environment"""
    mode: str = field(default="train")
    # Dataset config
    train_path: str = field(default="/rt-vepfs/wqs/workspace/RAGEN/data/webbrowser/train_admin_cleaned.parquet")
    val_path: str = field(default="/rt-vepfs/wqs/workspace/RAGEN/data/webbrowser/train_admin_cleaned.parquet")

    # env settings
    max_steps: int = field(default=100)

    # agent settings
    enable_visual: bool = field(default=False)
    high_level_action_set: Optional[List[str]] = field(default=None)
    strict: bool = field(default=False)
    multiaction: bool = field(default=True)
    
    # browsering settings
    search_engine: str = field(default="bing.com")
    browser_retries_limit: int = field(default=3)
    
    # 自动登录设置
    enable_auto_login: bool = field(default=True)
    auth_folder: str = field(default="./.auth")
    
    # WebArena URL环境变量映射（用于开发时）
    webarena_urls: Optional[Dict[str, str]] = field(default=None)
    
    def __post_init__(self):
        """初始化后处理"""
        if self.webarena_urls is None:
            self.webarena_urls = {
                "reddit": "",
                "shopping": "",
                "shopping_admin": "",
                "gitlab": "",
            }
    
    
