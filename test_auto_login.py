#!/usr/bin/env python3
"""
测试自动登录功能
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent))

from ragen.env.webbrowser.env import WebBrowserEnv, ACCOUNTS, URLS
from ragen.env.webbrowser.config import WebBrowserEnvConfig

def test_auto_login():
    """测试自动登录功能"""
    print("🧪 开始测试自动登录功能...")
    
    # 检查配置
    print(f"📋 ACCOUNTS配置: {list(ACCOUNTS.keys())}")
    print(f"📋 URLS配置: {list(URLS.keys())}")
    
    # 创建配置
    config = WebBrowserEnvConfig(
        enable_auto_login=True,
        auth_folder="./.auth"
    )
    
    print(f"🔧 配置: enable_auto_login={config.enable_auto_login}")
    print(f"🔧 配置: auth_folder={config.auth_folder}")
    
    try:
        # 创建环境实例
        print("🔧 创建WebBrowserEnv实例...")
        env = WebBrowserEnv(config=config)
        
        # 测试认证状态加载
        print("🔐 测试认证状态加载...")
        print(f"已加载的认证状态: {list(env._storage_states.keys())}")
        
        # 测试任务认证状态匹配
        print("🔍 测试任务认证状态匹配...")
        test_task = {
            "require_login": True,
            "sites": ["shopping_admin"],
            "storage_state": "shopping_admin_state.json"
        }
        
        storage_state_path = env._get_storage_state_for_task(test_task)
        print(f"匹配结果: {storage_state_path}")
        
        print("✅ 测试完成！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_auto_login() 