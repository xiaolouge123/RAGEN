import os
from typing import Dict
from dataclasses import dataclass, field
from ragen.env.webbrowser.config import WebBrowserEnvConfig

REDDIT = os.environ.get("REDDIT", "http://10.0.0.104:9999")
SHOPPING = os.environ.get("SHOPPING", "http://10.0.0.104:7770")
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "http://10.0.0.104:7780/admin")
GITLAB = os.environ.get("GITLAB", "http://10.0.0.104:8023")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "http://10.0.0.104:8888")

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

    url_mapping: dict = field(default_factory=lambda: {
        "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
        "__SHOPPING__": SHOPPING,
        "__REDDIT__": REDDIT,
        "__GITLAB__": GITLAB,
        "__WIKIPEDIA__": WIKIPEDIA,
    })