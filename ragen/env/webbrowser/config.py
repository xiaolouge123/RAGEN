from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List  

@dataclass
class WebBrowserEnvConfig:
    """Configuration for WebBrowserEnv environment"""
    mode: str = field(default="train")
    # Dataset config
    train_path: str = field(default="/rt-vepfs/zyc/workspaces/RAGEN/data/webbrowser/train.parquet")
    val_path: str = field(default="/rt-vepfs/zyc/workspaces/RAGEN/data/webbrowser/val.parquet")

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
    
    
