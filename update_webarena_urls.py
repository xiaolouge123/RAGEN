#!/usr/bin/env python3
"""
修改webarena_train.parquet文件中的URL映射
"""

import json
import pandas as pd
from pathlib import Path

def update_webarena_urls():
    """更新WebArena数据集中的URL映射"""
    
    # 新的URL映射
    URL_MAPPINGS = {
        "shopping": "http://10.0.0.104:7770",
        "shopping_admin": "http://10.0.0.104:7780", 
        "reddit": "http://10.0.0.104:9999"
    }
    
    # 网站到data_source的映射
    SITE_TO_DATA_SOURCE = {
        "shopping": "www.onestopmarket.com",
        "shopping_admin": "www.luma.com",
        "reddit": "www.reddit.com"
    }
    
    input_file = "data/webbrowser/webarena_train.parquet"
    output_file = "data/webbrowser/webarena_train_updated.parquet"
    
    print("🔄 开始更新WebArena数据集URL映射")
    print("="*50)
    
    try:
        # 读取数据
        print(f"📖 读取文件: {input_file}")
        df = pd.read_parquet(input_file)
        print(f"✅ 成功读取 {len(df)} 行数据")
        
        # 显示修改前的统计
        print("\n📊 修改前的URL分布:")
        print("-"*30)
        print("data_source分布:")
        print(df['data_source'].value_counts())
        print("\ndata_url分布:")
        print(df['data_url'].value_counts())
        print("\ntarget_url分布:")
        print(df['target_url'].value_counts())
        
        # 更新URL映射
        print("\n🔄 开始更新URL映射...")
        
        # 更新data_source
        for site, new_source in SITE_TO_DATA_SOURCE.items():
            mask = df['data_source'].str.contains(site, case=False, na=False)
            if mask.any():
                df.loc[mask, 'data_source'] = new_source
                print(f"  ✅ 更新 {site} 的data_source为: {new_source}")
        
        # 更新data_url和target_url - 修复匹配逻辑
        for site, new_url in URL_MAPPINGS.items():
            # 更新data_url - 使用更精确的匹配
            if site == "shopping":
                # 匹配包含"onestopmarket.com"的URL
                mask_data = df['data_url'].str.contains("onestopmarket.com", case=False, na=False)
            elif site == "shopping_admin":
                # 匹配包含"luma.com/admin"的URL
                mask_data = df['data_url'].str.contains("luma.com/admin", case=False, na=False)
            elif site == "reddit":
                # 匹配包含"reddit.com"的URL
                mask_data = df['data_url'].str.contains("reddit.com", case=False, na=False)
            else:
                mask_data = df['data_url'].str.contains(site, case=False, na=False)
            
            if mask_data.any():
                df.loc[mask_data, 'data_url'] = new_url
                print(f"  ✅ 更新 {site} 的data_url为: {new_url} (匹配到 {mask_data.sum()} 条记录)")
            
            # 更新target_url - 使用相同的匹配逻辑
            if site == "shopping":
                mask_target = df['target_url'].str.contains("onestopmarket.com", case=False, na=False)
            elif site == "shopping_admin":
                mask_target = df['target_url'].str.contains("luma.com/admin", case=False, na=False)
            elif site == "reddit":
                mask_target = df['target_url'].str.contains("reddit.com", case=False, na=False)
            else:
                mask_target = df['target_url'].str.contains(site, case=False, na=False)
            
            if mask_target.any():
                df.loc[mask_target, 'target_url'] = new_url
                print(f"  ✅ 更新 {site} 的target_url为: {new_url} (匹配到 {mask_target.sum()} 条记录)")
        
        # 显示修改后的统计
        print("\n📊 修改后的URL分布:")
        print("-"*30)
        print("data_source分布:")
        print(df['data_source'].value_counts())
        print("\ndata_url分布:")
        print(df['data_url'].value_counts())
        print("\ntarget_url分布:")
        print(df['target_url'].value_counts())
        
        # 保存更新后的文件
        print(f"\n💾 保存更新后的文件: {output_file}")
        df.to_parquet(output_file, index=False)
        print("✅ 文件保存成功!")
        
        # 验证更新结果
        print("\n🔍 验证更新结果:")
        print("-"*30)
        
        # 检查是否还有旧的URL
        old_urls = ["luma.com", "onestopmarket.com", "reddit.com"]
        for old_url in old_urls:
            data_url_count = df['data_url'].str.contains(old_url, case=False, na=False).sum()
            target_url_count = df['target_url'].str.contains(old_url, case=False, na=False).sum()
            if data_url_count > 0 or target_url_count > 0:
                print(f"⚠️  发现残留的旧URL: {old_url} (data_url: {data_url_count}, target_url: {target_url_count})")
            else:
                print(f"✅ 已完全替换旧URL: {old_url}")
        
        # 检查新URL是否正确设置
        new_urls = list(URL_MAPPINGS.values())
        for new_url in new_urls:
            data_url_count = df['data_url'].str.contains(new_url, case=False, na=False).sum()
            target_url_count = df['target_url'].str.contains(new_url, case=False, na=False).sum()
            print(f"✅ 新URL {new_url}: data_url={data_url_count}, target_url={target_url_count}")
        
        print(f"\n🎉 URL映射更新完成!")
        print(f"📁 新文件位置: {output_file}")
        print(f"📊 总行数: {len(df)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 更新失败: {e}")
        return False

if __name__ == "__main__":
    success = update_webarena_urls()
    if success:
        print("\n💡 提示: 如果更新结果满意，可以运行以下命令替换原文件:")
        print("mv data/webbrowser/webarena_train_updated.parquet data/webbrowser/webarena_train.parquet")
    else:
        print("\n❌ 更新失败，请检查错误信息") 