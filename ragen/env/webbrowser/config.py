from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List  

@dataclass
class WebBrowserEnvConfig:
    """Configuration for WebBrowserEnv environment"""
    # Dataset config
    dataset_name: str = field(default="web_crawler_stage_1")
    cache_dir: str = field(default="./data")
    split: Optional[str] = field(default=None)

    # env settings
    max_steps: int = field(default=100)
    enable_visual: bool = field(default=False)
    high_level_action_set: Optional[List[str]] = field(default=["bid", "nav"])
    strict: bool = field(default=False)
    multiaction: bool = field(default=True)
    
    # browsering settings
    search_engine: str = field(default="bing.com")
    
    
