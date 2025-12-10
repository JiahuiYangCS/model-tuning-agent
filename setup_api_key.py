#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Key Setup Script / API密钥设置脚本

用法 / Usage:
1. 将你的 OpenAI API Key 粘贴到下面的 API_KEY 变量中
   Paste your OpenAI API Key to the API_KEY variable below
2. 运行此脚本: python setup_api_key.py
   Run this script: python setup_api_key.py
3. 脚本将自动创建/更新 .env 文件，并将密钥写入
   The script will auto create/update .env file and write the key

示例 / Example:
    API_KEY = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    python setup_api_key.py
    → .env 文件已创建/更新 / .env file created/updated
"""

import os
import sys
from pathlib import Path

# =====================================================================
# 【重要】请在这里粘贴你的 OpenAI API Key
# 【IMPORTANT】Paste your OpenAI API Key here
# =====================================================================
API_KEY = ""

# =====================================================================


def setup_api_key(api_key):
    """
    将 API Key 写入 .env 文件
    Write API Key to .env file
    
    参数 / Args:
        api_key (str): OpenAI API Key
    
    返回 / Returns:
        bool: 成功返回 True，失败返回 False
    """
    
    if not api_key or not api_key.strip():
        print("❌ 错误 / Error: API Key 为空，请在脚本顶部粘贴你的 Key")
        print("❌ Error: API Key is empty. Please paste your key at the top of the script")
        return False
    
    api_key = api_key.strip()
    
    # 验证 API Key 格式 / Validate API Key format
    if not api_key.startswith("sk-"):
        print("⚠️  警告 / Warning: API Key 似乎不是有效的 OpenAI 格式 (应以 'sk-' 开头)")
        print("⚠️  Warning: API Key doesn't look like a valid OpenAI key (should start with 'sk-')")
        response = input("是否继续? / Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            return False
    
    # 获取项目根目录 / Get project root directory
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    
    try:
        # 读取现有的 .env 内容 / Read existing .env content
        existing_content = ""
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                existing_content = f.read()
        
        # 检查是否已存在 OPENAI_API_KEY / Check if OPENAI_API_KEY already exists
        if "OPENAI_API_KEY=" in existing_content:
            # 替换现有的 Key / Replace existing key
            lines = existing_content.split("\n")
            new_lines = []
            for line in lines:
                if line.startswith("OPENAI_API_KEY="):
                    new_lines.append(f"OPENAI_API_KEY={api_key}")
                else:
                    new_lines.append(line)
            new_content = "\n".join(new_lines)
        else:
            # 添加新的 Key / Add new key
            if existing_content and not existing_content.endswith("\n"):
                new_content = existing_content + "\n" + f"OPENAI_API_KEY={api_key}"
            else:
                new_content = existing_content + f"OPENAI_API_KEY={api_key}\n"
        
        # 写入 .env 文件 / Write to .env file
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print("✅ 成功 / Success!")
        print(f"✅ API Key 已写入: {env_file}")
        print(f"✅ API Key written to: {env_file}")
        print()
        print(f"🔑 Key: {api_key[:20]}...{api_key[-10:]}")  # 显示部分 Key / Show partial key
        print()
        print("💡 现在你可以运行主程序了:")
        print("   python agent_main_v6.py")
        print()
        print("💡 Now you can run the main program:")
        print("   python agent_main_v6.py")
        
        return True
    
    except Exception as e:
        print(f"❌ 错误 / Error: 写入 .env 文件失败")
        print(f"❌ Error: Failed to write .env file")
        print(f"   {str(e)}")
        return False


def main():
    """主函数 / Main function"""
    print("=" * 60)
    print("OpenAI API Key Setup / OpenAI API 密钥设置")
    print("=" * 60)
    print()
    
    if not API_KEY:
        print("❌ 请在脚本顶部的 API_KEY 变量中粘贴你的 API Key")
        print("❌ Please paste your API Key in the API_KEY variable at the top of the script")
        print()
        print("位置 / Location:")
        print("  API_KEY = \"\"  ← 在这里粘贴 / Paste here")
        print()
        sys.exit(1)
    
    success = setup_api_key(API_KEY)
    
    if not success:
        print()
        print("❌ 设置失败 / Setup failed")
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("✨ 设置完成 / Setup Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
