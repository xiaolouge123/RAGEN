#!/usr/bin/env python3
"""
从test.filtered.json构建符合RAGEN格式的train.parquet文件
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

def build_train_parquet():
    """从test.filtered.json构建train.parquet文件"""
    
    # 目标网站和对应的URL映射
    TARGET_SITES = ["shopping_admin", "shopping", "reddit"]
    URL_MAPPINGS = {
        "shopping": "http://10.0.0.104:7770",
        "shopping_admin": "http://10.0.0.104:7780", 
        "reddit": "http://10.0.0.104:9999"
    }
    
    # 网站到data_source的映射（使用真实URL）
    SITE_TO_DATA_SOURCE = {
        "shopping": "http://10.0.0.104:7770",
        "shopping_admin": "http://10.0.0.104:7780",
        "reddit": "http://10.0.0.104:9999"
    }
    
    input_file = "../webarena/config_files/test.filtered.json"
    output_file = "data/webbrowser/train.parquet"
    
    print("🔄 开始从test.filtered.json构建train.parquet")
    print("="*60)
    
    try:
        # 读取JSON文件
        print(f"📖 读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 成功读取 {len(data)} 条记录")
        
        # 过滤目标网站
        print(f"\n🔍 过滤目标网站: {TARGET_SITES}")
        filtered_data = []
        for item in data:
            sites = item.get("sites", [])
            if any(site in TARGET_SITES for site in sites):
                filtered_data.append(item)
        
        print(f"✅ 过滤后剩余 {len(filtered_data)} 条记录")
        
        # 显示各网站分布
        site_counts = {}
        for item in filtered_data:
            for site in item.get("sites", []):
                if site in TARGET_SITES:
                    site_counts[site] = site_counts.get(site, 0) + 1
                    break
        
        print("\n📊 各网站数据分布:")
        for site, count in site_counts.items():
            print(f"  {site}: {count} 条")
        
        # 转换为parquet格式
        print("\n🔄 转换为parquet格式...")
        parquet_data = []
        
        for item in filtered_data:
            sites = item.get("sites", [])
            target_site = None
            
            # 找到目标网站
            for site in sites:
                if site in TARGET_SITES:
                    target_site = site
                    break
            
            if not target_site:
                continue
            
            # 解析start_url
            start_url = item.get("start_url", "")
            data_url = URL_MAPPINGS[target_site]
            target_url = URL_MAPPINGS[target_site]
            data_source = SITE_TO_DATA_SOURCE[target_site]
            
            # 提取instruction和ground_truth
            instruction = item.get("intent", "")
            ground_truth = ""
            
            # 从eval中提取ground_truth
            eval_data = item.get("eval", {})
            if eval_data:
                ref_answers = eval_data.get("reference_answers", {})
                if ref_answers:
                    # 优先使用reference_answer_raw_annotation
                    ground_truth = eval_data.get("reference_answer_raw_annotation", "")
                    if not ground_truth:
                        # 如果没有raw_annotation，尝试其他字段
                        if ref_answers.get("exact_match"):
                            ground_truth = ref_answers["exact_match"]
                        elif ref_answers.get("must_include"):
                            ground_truth = ref_answers["must_include"]
                        elif ref_answers.get("fuzzy_match"):
                            ground_truth = ref_answers["fuzzy_match"]
            
            # 构建parquet记录
            parquet_record = {
                "data_source": data_source,
                "data_url": data_url,
                "instruction": instruction,
                "action_tip": "",  # 空字符串
                "ground_truth": ground_truth,
                "target_url": target_url,
                # 保留WebArena的额外字段用于自动登录
                "task_id": item.get("task_id", 0),
                "require_login": item.get("require_login", False),
                "storage_state": item.get("storage_state", ""),
                "sites": sites,
                "eval_types": eval_data.get("eval_types", []),
                "intent_template": item.get("intent_template", ""),
                "intent_template_id": item.get("intent_template_id", 0)
            }
            
            parquet_data.append(parquet_record)
        
        print(f"✅ 成功转换 {len(parquet_data)} 条记录")
        
        # 创建DataFrame并保存
        print(f"\n💾 保存到: {output_file}")
        df = pd.DataFrame(parquet_data)
        df.to_parquet(output_file, index=False)
        print("✅ 文件保存成功!")
        
        # 验证结果
        print("\n📊 最终数据统计:")
        print("-"*40)
        print(f"总行数: {len(df)}")
        print(f"总列数: {len(df.columns)}")
        print(f"字段: {list(df.columns)}")
        
        print("\n🔗 URL分布:")
        print("-"*40)
        print("data_url分布:")
        print(df['data_url'].value_counts())
        print("\ntarget_url分布:")
        print(df['target_url'].value_counts())
        
        print("\n🏷️  data_source分布:")
        print("-"*40)
        print(df['data_source'].value_counts())
        
        # 检查自动登录相关字段
        print("\n🔐 自动登录相关字段:")
        print("-"*40)
        require_login_count = df['require_login'].sum()
        print(f"需要登录的任务: {require_login_count} 条")
        print(f"不需要登录的任务: {len(df) - require_login_count} 条")
        
        # 检查storage_state分布
        storage_states = df['storage_state'].value_counts()
        print(f"\n认证状态文件分布:")
        for state, count in storage_states.items():
            print(f"  {state}: {count} 条")
        
        print(f"\n🎉 train.parquet构建完成!")
        print(f"📁 文件位置: {output_file}")
        print(f"📊 总行数: {len(df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 构建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = build_train_parquet()
    if success:
        print("\n💡 提示:")
        print("1. 新构建的train.parquet已包含所有必需字段")
        print("2. URL已映射到指定的真实地址")
        print("3. 自动登录相关字段已保留")
        print("4. 可以在RAGEN配置中启用自动登录功能")
    else:
        print("\n❌ 构建失败，请检查错误信息") 