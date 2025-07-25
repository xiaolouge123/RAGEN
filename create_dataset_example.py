#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebArena数据集构造示例
演示如何从WebArena配置文件构造RAGEN格式的数据集
"""

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent))

from create_webarena_dataset import WebArenaDatasetCreator


def main():
    print("🚀 WebArena数据集构造示例")
    print("="*60)
    
    # 1. 设置环境变量（示例）
    # 注意：请根据您的实际WebArena服务器地址修改
    os.environ.update({
        "REDDIT": "http://your-reddit-server:9999",
        "SHOPPING": "http://your-shopping-server:7770", 
        "SHOPPING_ADMIN": "http://your-shopping-admin-server:7780/admin",
        "GITLAB": "http://your-gitlab-server:8023",
    })
    
    # 2. 创建数据集构造器
    creator = WebArenaDatasetCreator()
    
    # 3. 设置输入输出路径
    config_file = "webarena/config_files/test.filtered.json"  # WebArena配置文件
    output_file = "data/webbrowser/webarena_dataset.parquet"  # 输出parquet文件
    auth_folder = ".auth"  # 认证文件目录
    
    # 4. 检查输入文件是否存在
    if not Path(config_file).exists():
        print(f"❌ 配置文件不存在: {config_file}")
        print("请确保已下载WebArena项目和配置文件")
        return
    
    # 5. 创建输出目录
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 6. 创建数据集（不创建认证文件）
    print("\n📦 开始创建数据集...")
    creator.create_dataset(
        config_file=config_file,
        output_path=output_file,
        create_auth=False,  # 暂时不创建认证文件
        auth_folder=auth_folder
    )
    
    # 7. 验证输出文件
    if Path(output_file).exists():
        print(f"\n✅ 数据集创建成功!")
        print(f"输出文件: {output_file}")
        print(f"文件大小: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")
        
        # 验证文件内容
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(output_file)
            print(f"数据行数: {table.num_rows}")
            print(f"数据列数: {table.num_columns}")
            print(f"列名: {table.column_names}")
        except Exception as e:
            print(f"⚠️ 验证文件内容时出错: {e}")
    else:
        print(f"\n❌ 数据集创建失败!")
        return
    
    # 8. 可选：创建认证文件
    create_auth = input("\n是否创建认证文件? (y/N): ").lower().strip() == 'y'
    if create_auth:
        print("\n🔐 开始创建认证文件...")
        try:
            creator.create_auth_files(auth_folder)
            print(f"✅ 认证文件创建完成，保存在: {auth_folder}")
        except Exception as e:
            print(f"❌ 认证文件创建失败: {e}")
    
    print("\n🎉 示例运行完成!")
    print("\n📋 下一步:")
    print("1. 检查生成的parquet文件是否符合预期")
    print("2. 如果需要认证，设置环境变量并创建认证文件")
    print("3. 在RAGEN配置中指定数据集路径和认证设置")


def demo_ragen_integration():
    """演示如何在RAGEN中使用创建的数据集"""
    print("\n" + "="*60)
    print("🔗 RAGEN集成示例")
    print("="*60)
    
    try:
        from ragen.env.webbrowser.config import WebBrowserEnvConfig
        from ragen.env.webbrowser.env import WebBrowserEnv
    except ImportError as e:
        print(f"❌ 无法导入RAGEN模块: {e}")
        print("请确保您在RAGEN项目目录中运行此脚本")
        return
    
    # 创建配置
    config = WebBrowserEnvConfig(
        mode="train",
        train_path="data/webbrowser/webarena_dataset.parquet",
        enable_auto_login=True,  # 启用自动登录
        auth_folder=".auth",
        webarena_urls={
            "reddit": os.environ.get("REDDIT", ""),
            "shopping": os.environ.get("SHOPPING", ""),
            "shopping_admin": os.environ.get("SHOPPING_ADMIN", ""),
            "gitlab": os.environ.get("GITLAB", ""),
        }
    )
    
    print("配置示例:")
    print(f"  数据集路径: {config.train_path}")
    print(f"  自动登录: {config.enable_auto_login}")
    print(f"  认证文件夹: {config.auth_folder}")
    print(f"  网站URL配置: {config.webarena_urls}")
    
    # 注意：实际使用时需要确保数据集文件存在
    if Path(config.train_path).exists():
        print(f"\n✅ 数据集文件已就绪，可以开始训练!")
        
        # 可以创建环境进行测试
        try:
            env = WebBrowserEnv(config=config)
            print(f"✅ WebBrowserEnv环境创建成功!")
            
            # 测试重置
            obs = env.reset(seed=42)
            print(f"✅ 环境重置成功，观察内容长度: {len(obs.content)}")
            
            env.close()
            print(f"✅ 环境关闭成功!")
            
        except Exception as e:
            print(f"⚠️ 环境测试遇到问题: {e}")
    else:
        print(f"\n⚠️ 数据集文件不存在: {config.train_path}")
        print("请先运行数据集创建脚本")


def test_with_existing_config():
    """使用现有的test.filtered.json测试"""
    print("\n" + "="*60)
    print("🧪 使用现有配置文件测试")
    print("="*60)
    
    # 查找可能的配置文件位置
    possible_configs = [
        "../webarena/config_files/test.filtered.json",
        "webarena/config_files/test.filtered.json",
        "./config_files/test.filtered.json",
    ]
    
    config_file = None
    for path in possible_configs:
        if Path(path).exists():
            config_file = path
            break
    
    if not config_file:
        print("❌ 未找到test.filtered.json配置文件")
        print("请确保WebArena项目在正确位置，或手动指定配置文件路径")
        return
    
    print(f"✅ 找到配置文件: {config_file}")
    
    # 创建数据集构造器并测试
    creator = WebArenaDatasetCreator()
    
    # 只加载和分析配置文件，不保存
    try:
        data = creator.load_config_data(config_file)
        filtered_data = creator.filter_target_sites(data)
        
        print(f"\n📊 数据分析结果:")
        print(f"原始数据条数: {len(data)}")
        print(f"目标网站筛选后: {len(filtered_data)}")
        
        # 分析网站分布
        site_counts = {}
        for item in filtered_data:
            sites = item.get("sites", [])
            for site in sites:
                if site in creator.TARGET_SITES:
                    site_counts[site] = site_counts.get(site, 0) + 1
        
        print(f"网站分布:")
        for site, count in site_counts.items():
            print(f"  {site}: {count}")
        
        # 分析需要登录的任务
        login_required = sum(1 for item in filtered_data if item.get("require_login", False))
        print(f"需要登录的任务: {login_required}/{len(filtered_data)}")
        
        print(f"\n✅ 配置文件分析完成!")
        
    except Exception as e:
        print(f"❌ 分析配置文件时出错: {e}")


if __name__ == "__main__":
    main()
    
    # 可选：演示RAGEN集成
    demo_integration = input("\n是否演示RAGEN集成? (y/N): ").lower().strip() == 'y'
    if demo_integration:
        demo_ragen_integration()
    
    # 可选：测试现有配置文件
    test_existing = input("\n是否测试现有配置文件? (y/N): ").lower().strip() == 'y'
    if test_existing:
        test_with_existing_config() 