import os
from typing import Dict
from dataclasses import dataclass, field
from ragen.env.webbrowser.config import WebBrowserEnvConfig

REDDIT = os.environ.get("REDDIT", "")
SHOPPING = os.environ.get("SHOPPING", "")
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")
GITLAB = os.environ.get("GITLAB", "")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "")

ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
}

@dataclass
class WebArenaBrowserEnvConfig(WebBrowserEnvConfig):
    """Configuration for WebArenaBrowserEnv environment"""
    # Dataset config
    train_path: str = field(default="/rt-vepfs/zyc/workspaces/RAGEN/data/webarena/webarena.train.parquet")
    val_path: str = field(default="/rt-vepfs/zyc/workspaces/RAGEN/data/webarena/webarena.val.parquet")

    # login state dir
    storage_state_dir: str = field(default="/rt-vepfs/zyc/workspaces/RAGEN/data/webarena/auth")

    url_mapping: Dict[str, str] = {
        "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
        "__SHOPPING__": SHOPPING,
        "__REDDIT__": REDDIT,
        "__GITLAB__": GITLAB,
        "__WIKIPEDIA__": WIKIPEDIA,
    }