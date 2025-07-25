#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WebArena数据集构造工具
从配置文件中提取问题、答案和网站信息，生成parquet格式的数据集
同时提供自动登录功能
"""

import json
import os
import time
import argparse
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
import glob
import sys

# 导入WebArena的环境配置
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("请安装playwright: pip install playwright")
    exit(1)

# 尝试导入WebArena的环境配置
try:
    sys.path.append(str(Path(__file__).parent.parent / "webarena"))
    from browser_env.env_config import ACCOUNTS, URL_MAPPINGS
    from browser_env.env_config import REDDIT, SHOPPING, SHOPPING_ADMIN, GITLAB, WIKIPEDIA, MAP, HOMEPAGE
    WEBARENA_AVAILABLE = True
    print("✅ 成功导入WebArena环境配置")
except ImportError as e:
    print(f"⚠️ 无法导入WebArena环境配置: {e}")
    print("将使用默认配置")
    WEBARENA_AVAILABLE = False
    
    # 默认配置
    ACCOUNTS = {
        "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
        "gitlab": {"username": "byteblaze", "password": "hello1234"},
        "shopping": {"username": "emma.lopez@gmail.com", "password": "Password.123"},
        "shopping_admin": {"username": "admin", "password": "admin1234"},
    }
    
    # 从环境变量获取URL
    REDDIT = os.environ.get("REDDIT", "")
    SHOPPING = os.environ.get("SHOPPING", "")
    SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "")
    GITLAB = os.environ.get("GITLAB", "")
    WIKIPEDIA = os.environ.get("WIKIPEDIA", "")
    MAP = os.environ.get("MAP", "")
    HOMEPAGE = os.environ.get("HOMEPAGE", "")
    
    URL_MAPPINGS = {
        REDDIT: "http://reddit.com",
        SHOPPING: "http://onestopmarket.com", 
        SHOPPING_ADMIN: "http://luma.com/admin",
        GITLAB: "http://gitlab.com",
        WIKIPEDIA: "http://wikipedia.org",
        MAP: "http://openstreetmap.org",
        HOMEPAGE: "http://homepage.com",
    }


class WebArenaDatasetCreator:
    """WebArena数据集创建器"""
    
    def __init__(self):
        # 设置支持的网站
        self.TARGET_SITES = ["shopping_admin", "shopping", "reddit"]
        
        # WebArena环境URL（从环境变量获取）
        self.WEBARENA_URLS = {
            "reddit": REDDIT,
            "shopping": SHOPPING,
            "shopping_admin": SHOPPING_ADMIN,
            "gitlab": GITLAB,
            "wikipedia": WIKIPEDIA,
            "map": MAP,
            "homepage": HOMEPAGE,
        }
        
        # 标准化URL映射（WebArena的URL_MAPPINGS）
        self.URL_MAPPINGS = URL_MAPPINGS
        
        # 反向映射：从标准化URL到环境URL
        self.REVERSE_URL_MAPPINGS = {v: k for k, v in URL_MAPPINGS.items() if k}
        
        # 模板到环境变量的映射
        self.TEMPLATE_TO_ENV = {
            "__SHOPPING_ADMIN__": SHOPPING_ADMIN,
            "__SHOPPING__": SHOPPING,
            "__REDDIT__": REDDIT,
            "__GITLAB__": GITLAB,
            "__WIKIPEDIA__": WIKIPEDIA,
            "__MAP__": MAP,
            "__HOMEPAGE__": HOMEPAGE,
        }
        
        # 模板到标准化URL的映射
        self.TEMPLATE_TO_STANDARD = {
            "__SHOPPING_ADMIN__": "http://10.0.0.104:7780",
            "__SHOPPING__": "http://10.0.0.104:7770",
            "__REDDIT__": "http://10.0.0.104:9999",
            "__GITLAB__": "http://gitlab.com",
            "__WIKIPEDIA__": "http://wikipedia.org",
            "__MAP__": "http://openstreetmap.org",
            "__HOMEPAGE__": "http://homepage.com",
        }
        
        # 网站到data_source的映射
        self.SITE_TO_DATA_SOURCE = {
            "shopping_admin": "www.luma.com",
            "shopping": "www.onestopmarket.com", 
            "reddit": "www.reddit.com",
            "gitlab": "www.gitlab.com",
        }
        
        # 默认账户信息
        self.ACCOUNTS = ACCOUNTS
        
        # 自动登录相关配置
        self.SITES = ["gitlab", "shopping", "shopping_admin", "reddit"]
        self.EXACT_MATCH = [True, True, True, True]
        self.KEYWORDS = ["", "", "Dashboard", "Delete"]
    
    def load_config_data(self, config_file: str) -> List[Dict[str, Any]]:
        """加载配置文件数据"""
        print(f"正在读取配置文件: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"成功读取 {len(data)} 条配置数据")
        return data
    
    def filter_target_sites(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """筛选目标网站的数据"""
        filtered_data = []
        
        for item in data:
            sites = item.get("sites", [])
            # 检查是否包含目标网站
            if any(site in self.TARGET_SITES for site in sites):
                filtered_data.append(item)
        
        print(f"筛选后保留 {len(filtered_data)} 条数据 (目标网站: {self.TARGET_SITES})")
        return filtered_data
    
    def resolve_start_url(self, start_url: str, sites: List[str]) -> tuple[str, str, str]:
        """
        解析start_url模板为真实URL
        
        Returns:
            tuple[data_source, data_url, target_url]
        """
        # 默认data_source
        data_source = sites[0] if sites else "unknown"
        if data_source in self.SITE_TO_DATA_SOURCE:
            data_source = self.SITE_TO_DATA_SOURCE[data_source]
        
        # 检查是否为模板URL
        for template, env_url in self.TEMPLATE_TO_ENV.items():
            if template in start_url:
                # 获取标准化URL
                standard_url = self.TEMPLATE_TO_STANDARD.get(template, template)
                
                if env_url:  # 如果环境变量已设置，使用环境URL
                    data_url = start_url.replace(template, env_url)
                    target_url = data_url
                else:  # 如果环境变量未设置，使用标准化URL
                    data_url = start_url.replace(template, standard_url)
                    target_url = data_url
                
                return data_source, data_url, target_url
        
        # 如果不是模板URL，直接返回
        return data_source, start_url, start_url
    
    def extract_answer(self, eval_data: Dict[str, Any]) -> str:
        """从eval数据中提取答案"""
        # 优先使用reference_answer_raw_annotation
        if "reference_answer_raw_annotation" in eval_data:
            raw_annotation = eval_data["reference_answer_raw_annotation"]
            if raw_annotation is not None:
                return str(raw_annotation)
        
        # 如果没有，尝试从reference_answers中提取
        ref_answers = eval_data.get("reference_answers")
        if ref_answers is None:
            return ""
        
        if "exact_match" in ref_answers:
            exact_match = ref_answers["exact_match"]
            if exact_match is not None:
                return str(exact_match)
        elif "must_include" in ref_answers:
            must_include = ref_answers["must_include"]
            if must_include is not None:
                if isinstance(must_include, list):
                    return "; ".join(str(x) for x in must_include if x is not None)
                else:
                    return str(must_include)
        elif "fuzzy_match" in ref_answers:
            fuzzy_match = ref_answers["fuzzy_match"]
            if fuzzy_match is not None:
                if isinstance(fuzzy_match, list):
                    return "; ".join(str(x) for x in fuzzy_match if x is not None)
                else:
                    return str(fuzzy_match)
        
        return ""
    
    def convert_to_parquet_format(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将WebArena配置数据转换为parquet格式"""
        converted_data = []
        
        for item in data:
            # 解析URL
            start_url = item.get("start_url", "")
            sites = item.get("sites", [])
            data_source, data_url, target_url = self.resolve_start_url(start_url, sites)
            
            # 提取答案
            eval_data = item.get("eval", {})
            ground_truth = self.extract_answer(eval_data)
            
            # 构造parquet格式的数据
            parquet_item = {
                "data_source": data_source,
                "data_url": data_url,
                "instruction": item.get("intent", ""),
                "action_tip": "",  # 按要求设置为空字符串
                "ground_truth": ground_truth,
                "target_url": target_url,
                # 额外保留的WebArena字段（用于调试）
                "task_id": item.get("task_id", -1),
                "require_login": item.get("require_login", False),
                "storage_state": item.get("storage_state", ""),
                "sites": item.get("sites", []),
                "eval_types": eval_data.get("eval_types", []),
                "intent_template": item.get("intent_template", ""),
                "intent_template_id": item.get("intent_template_id", -1),
            }
            
            converted_data.append(parquet_item)
        
        return converted_data
    
    def save_to_parquet(self, data: List[Dict[str, Any]], output_path: str):
        """保存数据为parquet格式"""
        print(f"正在保存数据到: {output_path}")
        
        # 确保输出目录存在
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 创建PyArrow表
        if not data:
            print("警告: 没有数据可保存")
            return
        
        # 定义schema
        schema = pa.schema([
            ('data_source', pa.string()),
            ('data_url', pa.string()),
            ('instruction', pa.string()),
            ('action_tip', pa.string()),
            ('ground_truth', pa.string()),
            ('target_url', pa.string()),
            ('task_id', pa.int64()),
            ('require_login', pa.bool_()),
            ('storage_state', pa.string()),
            ('sites', pa.list_(pa.string())),
            ('eval_types', pa.list_(pa.string())),
            ('intent_template', pa.string()),
            ('intent_template_id', pa.int64()),
        ])
        
        # 转换数据
        arrays = []
        for field in schema:
            field_name = field.name
            field_data = [item[field_name] for item in data]
            arrays.append(pa.array(field_data))
        
        # 创建表
        table = pa.table(arrays, schema=schema)
        
        # 保存为parquet
        pq.write_table(table, output_path)
        
        print(f"成功保存 {len(data)} 条数据到 {output_path}")
        
        # 打印数据统计
        print("\n数据统计:")
        print(f"总条数: {len(data)}")
        print(f"网站分布:")
        site_counts = {}
        for item in data:
            sites = item.get("sites", [])
            for site in sites:
                if site in self.TARGET_SITES:
                    site_counts[site] = site_counts.get(site, 0) + 1
        
        for site, count in site_counts.items():
            print(f"  {site}: {count}")
        print(f"需要登录的任务数: {sum(1 for item in data if item['require_login'])}")
        
        # 显示URL映射示例
        print(f"\nURL映射示例:")
        for i, item in enumerate(data[:3]):
            print(f"  示例 {i+1}:")
            print(f"    data_source: {item['data_source']}")
            print(f"    data_url: {item['data_url']}")
            print(f"    target_url: {item['target_url']}")
    
    # ========== 自动登录功能 ==========
    
    def is_expired(self, storage_state: Path, url: str, keyword: str, url_exact: bool = True) -> bool:
        """测试cookie是否过期"""
        if not storage_state.exists():
            return True

        context_manager = sync_playwright()
        playwright = context_manager.__enter__()
        browser = playwright.chromium.launch(headless=True, slow_mo=0)
        context = browser.new_context(storage_state=storage_state)
        page = context.new_page()
        page.goto(url)
        time.sleep(1)
        d_url = page.url
        content = page.content()
        context_manager.__exit__()
        
        if keyword:
            return keyword not in content
        else:
            if url_exact:
                return d_url != url
            else:
                return url not in d_url
    
    def renew_comb(self, comb: list[str], auth_folder: str = "./.auth") -> None:
        """为指定的网站组合更新认证状态"""
        print(f"正在为 {comb} 更新认证状态...")
        
        context_manager = sync_playwright()
        playwright = context_manager.__enter__()
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        if "shopping" in comb:
            username = self.ACCOUNTS["shopping"]["username"]
            password = self.ACCOUNTS["shopping"]["password"]
            shopping_url = self.WEBARENA_URLS.get("shopping", "")
            if shopping_url:
                page.goto(f"{shopping_url}/customer/account/login/")
                page.get_by_label("Email", exact=True).fill(username)
                page.get_by_label("Password", exact=True).fill(password)
                page.get_by_role("button", name="Sign In").click()
                print(f"  ✓ Shopping网站登录完成")

        if "reddit" in comb:
            username = self.ACCOUNTS["reddit"]["username"]
            password = self.ACCOUNTS["reddit"]["password"]
            reddit_url = self.WEBARENA_URLS.get("reddit", "")
            if reddit_url:
                page.goto(f"{reddit_url}/login")
                page.get_by_label("Username").fill(username)
                page.get_by_label("Password").fill(password)
                page.get_by_role("button", name="Log in").click()
                print(f"  ✓ Reddit网站登录完成")

        if "shopping_admin" in comb:
            username = self.ACCOUNTS["shopping_admin"]["username"]
            password = self.ACCOUNTS["shopping_admin"]["password"]
            admin_url = self.WEBARENA_URLS.get("shopping_admin", "")
            if admin_url:
                page.goto(admin_url)
                page.get_by_placeholder("user name").fill(username)
                page.get_by_placeholder("password").fill(password)
                page.get_by_role("button", name="Sign in").click()
                print(f"  ✓ Shopping Admin网站登录完成")

        if "gitlab" in comb:
            username = self.ACCOUNTS["gitlab"]["username"]
            password = self.ACCOUNTS["gitlab"]["password"]
            gitlab_url = self.WEBARENA_URLS.get("gitlab", "")
            if gitlab_url:
                page.goto(f"{gitlab_url}/users/sign_in")
                page.get_by_test_id("username-field").click()
                page.get_by_test_id("username-field").fill(username)
                page.get_by_test_id("username-field").press("Tab")
                page.get_by_test_id("password-field").fill(password)
                page.get_by_test_id("sign-in-button").click()
                print(f"  ✓ GitLab网站登录完成")

        # 保存认证状态
        os.makedirs(auth_folder, exist_ok=True)
        auth_file = f"{auth_folder}/{'.'.join(comb)}_state.json"
        context.storage_state(path=auth_file)
        print(f"  ✓ 认证状态已保存到: {auth_file}")
        
        context_manager.__exit__()
    
    def get_site_comb_from_filepath(self, file_path: str) -> list[str]:
        """从文件路径中提取网站组合"""
        comb = os.path.basename(file_path).rsplit("_", 1)[0].split(".")
        return comb
    
    def create_auth_files(self, auth_folder: str = "./.auth") -> None:
        """创建所有需要的认证文件"""
        print("开始创建认证文件...")
        
        # 检查环境变量
        missing_urls = []
        for site in self.TARGET_SITES:
            env_key = site.upper()
            if not self.WEBARENA_URLS.get(site):
                missing_urls.append(env_key)
        
        if missing_urls:
            print(f"警告: 缺少以下环境变量: {missing_urls}")
            print("请设置对应的环境变量，例如:")
            for site in missing_urls:
                print(f"export {site}='http://your-{site.lower().replace('_', '-')}-domain'")
            print()
        
        # 为目标网站创建认证文件
        target_sites_filtered = [site for site in self.TARGET_SITES if self.WEBARENA_URLS.get(site)]
        
        # 单个网站的认证
        for site in target_sites_filtered:
            try:
                self.renew_comb([site], auth_folder=auth_folder)
            except Exception as e:
                print(f"创建 {site} 认证文件失败: {e}")
        
        # 网站组合的认证
        pairs = list(combinations(target_sites_filtered, 2))
        for pair in pairs:
            # 跳过某些不兼容的组合
            if "reddit" in pair and ("shopping" in pair or "shopping_admin" in pair):
                continue
            try:
                self.renew_comb(list(sorted(pair)), auth_folder=auth_folder)
            except Exception as e:
                print(f"创建 {pair} 组合认证文件失败: {e}")
        
        print("认证文件创建完成!")
    
    def create_dataset(self, config_file: str, output_path: str, create_auth: bool = False, auth_folder: str = "./.auth"):
        """创建完整的数据集"""
        print("="*60)
        print("WebArena数据集构造工具")
        print("="*60)
        
        # 显示环境配置
        print(f"WebArena配置状态: {'✅ 可用' if WEBARENA_AVAILABLE else '⚠️ 不可用'}")
        print(f"目标网站: {self.TARGET_SITES}")
        print(f"环境URL配置:")
        for site in self.TARGET_SITES:
            url = self.WEBARENA_URLS.get(site, "")
            status = "✅" if url else "❌"
            print(f"  {status} {site.upper()}: {url or '未设置'}")
        print()
        
        # 1. 读取配置数据
        data = self.load_config_data(config_file)
        
        # 2. 筛选目标网站
        filtered_data = self.filter_target_sites(data)
        
        # 3. 转换格式
        parquet_data = self.convert_to_parquet_format(filtered_data)
        
        # 4. 保存parquet文件
        self.save_to_parquet(parquet_data, output_path)
        
        # 5. 创建认证文件（如果需要）
        if create_auth:
            self.create_auth_files(auth_folder)
        
        print("="*60)
        print("数据集构造完成!")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description="WebArena数据集构造工具")
    parser.add_argument("--config", type=str, required=True, help="WebArena配置文件路径")
    parser.add_argument("--output", type=str, required=True, help="输出parquet文件路径")
    parser.add_argument("--create-auth", action="store_true", help="是否创建认证文件")
    parser.add_argument("--auth-folder", type=str, default="./.auth", help="认证文件保存目录")
    parser.add_argument("--sites", nargs="+", default=["shopping_admin", "shopping", "reddit"], 
                       help="目标网站列表")
    
    args = parser.parse_args()
    
    # 创建数据集构造器
    creator = WebArenaDatasetCreator()
    creator.TARGET_SITES = args.sites
    
    # 创建数据集
    creator.create_dataset(
        config_file=args.config,
        output_path=args.output,
        create_auth=args.create_auth,
        auth_folder=args.auth_folder
    )


if __name__ == "__main__":
    main() 