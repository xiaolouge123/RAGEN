#!/usr/bin/env python3
"""
演示如何使用新的train.parquet数据集和自动登录功能
"""

import os
import sys
from pathlib import Path

# 添加RAGEN路径
sys.path.append('/rt-vepfs/wqs/workspace/RAGEN')

from ragen.env.webbrowser.config import WebBrowserEnvConfig
from ragen.env.webbrowser.env import WebBrowserEnv

def demo_auto_login():
    """演示自动登录功能"""
    
    print("🚀 演示RAGEN自动登录功能")
    print("="*50)
    
    # 配置WebBrowserEnv
    config = WebBrowserEnvConfig(
        mode="train",
        train_path="/rt-vepfs/wqs/workspace/RAGEN/data/webbrowser/train.parquet",
        val_path="/rt-vepfs/wqs/workspace/RAGEN/data/webbrowser/train.parquet",
        enable_auto_login=True,  # 启用自动登录
        auth_folder="./.auth",   # 认证文件目录
        max_steps=50,
        browser_retries_limit=3
    )
    
    print("📋 配置信息:")
    print(f"  数据集路径: {config.train_path}")
    print(f"  自动登录: {'启用' if config.enable_auto_login else '禁用'}")
    print(f"  认证目录: {config.auth_folder}")
    print(f"  最大步数: {config.max_steps}")
    
    # 检查认证文件目录
    auth_path = Path(config.auth_folder)
    if auth_path.exists():
        print(f"\n🔐 认证文件目录存在: {auth_path}")
        auth_files = list(auth_path.glob("*.json"))
        print(f"  找到 {len(auth_files)} 个认证文件:")
        for auth_file in auth_files:
            print(f"    - {auth_file.name}")
    else:
        print(f"\n⚠️  认证文件目录不存在: {auth_path}")
        print("  需要先运行认证脚本生成认证文件")
    
    try:
        # 创建环境
        print(f"\n🔧 创建WebBrowserEnv环境...")
        env = WebBrowserEnv(config=config)
        print("✅ 环境创建成功!")
        
        # 重置环境（会触发自动登录）
        print(f"\n🔄 重置环境...")
        obs = env.reset(seed=42)
        print("✅ 环境重置成功!")
        
        # 显示当前任务信息
        if hasattr(env, 'current_task') and env.current_task:
            task = env.current_task
            print(f"\n📋 当前任务信息:")
            print(f"  数据源: {task.data_source}")
            print(f"  数据URL: {task.data_url}")
            print(f"  目标URL: {task.target_url}")
            print(f"  指令: {task.instruction[:100]}...")
            print(f"  答案: {task.ground_truth}")
        
        # 显示观察信息
        print(f"\n👁️  观察信息:")
        print(f"  当前URL: {obs.url}")
        print(f"  内容长度: {len(obs.content)} 字符")
        print(f"  是否有错误: {obs.error}")
        
        # 执行一个简单的动作
        print(f"\n🎮 执行测试动作...")
        action = "goto('http://10.0.0.104:7770')"  # 访问shopping网站
        obs, reward, done, info = env.step(action)
        print(f"✅ 动作执行完成!")
        print(f"  奖励: {reward}")
        print(f"  是否结束: {done}")
        print(f"  新URL: {obs.url}")
        
        # 关闭环境
        print(f"\n🔚 关闭环境...")
        env.close()
        print("✅ 环境已关闭")
        
        print(f"\n🎉 演示完成!")
        print("💡 提示:")
        print("1. 自动登录功能已成功集成")
        print("2. 环境会根据任务自动选择合适的认证状态")
        print("3. 所有URL都已映射到指定的真实地址")
        print("4. 可以开始训练或测试RAGEN模型")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

def check_dataset():
    """检查数据集信息"""
    
    print("📊 检查数据集信息")
    print("="*30)
    
    try:
        import pandas as pd
        
        # 读取数据集
        df = pd.read_parquet("/rt-vepfs/wqs/workspace/RAGEN/data/webbrowser/train.parquet")
        
        print(f"总行数: {len(df)}")
        print(f"总列数: {len(df.columns)}")
        
        print("\n📋 字段列表:")
        for i, col in enumerate(df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\n🔗 URL分布:")
        print(df['data_url'].value_counts())
        
        print("\n🔐 登录需求:")
        print(f"需要登录: {df['require_login'].sum()} 条")
        print(f"不需要登录: {len(df) - df['require_login'].sum()} 条")
        
        print("\n📝 示例任务:")
        sample = df.iloc[0]
        print(f"指令: {sample['instruction']}")
        print(f"答案: {sample['ground_truth']}")
        print(f"数据源: {sample['data_source']}")
        print(f"数据URL: {sample['data_url']}")
        
    except Exception as e:
        print(f"❌ 检查数据集时出现错误: {e}")

if __name__ == "__main__":
    print("选择要运行的演示:")
    print("1. 检查数据集信息")
    print("2. 演示自动登录功能")
    
    choice = input("请输入选择 (1 或 2): ").strip()
    
    if choice == "1":
        check_dataset()
    elif choice == "2":
        demo_auto_login()
    else:
        print("无效选择，运行数据集检查...")
        check_dataset() 