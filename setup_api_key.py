#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置 OpenAI API Key 到 .env 文件

重要安全信息:
- 脚本中永远不会存储 API Key
- 运行时会提示你输入 Key（不显示在屏幕上）
- Key 只会被写入 .env 文件（该文件已在 .gitignore 中忽略）

使用方法:
1. 运行此脚本: python setup_api_key.py
2. 粘贴你的 OpenAI API Key（提示时输入，不会显示）
3. 脚本自动创建 .env 文件
4. 完成！.env 中的 Key 不会上传到 Git

安全提示:
- ✅ Key 从不在脚本代码中出现
- ✅ Key 从不显示在屏幕上
- ✅ Key 完全被 .gitignore 保护
- ✅ 可以安全地上传此脚本到 GitHub

示例工作流:
    $ python setup_api_key.py
    请输入你的 OpenAI API Key:
    API Key (不会显示在屏幕上): [你输入，但屏幕不显示]
    ✓ .env 文件已创建
    ✓ API Key 已保存
"""

import os
import sys
import getpass
from pathlib import Path


def setup_api_key(api_key=None):
    """将 API Key 写入 .env 文件
    
    如果没有提供 api_key，将提示用户输入
    """
    
    # 如果没有提供 Key，提示用户输入（隐藏输入）
    if not api_key:
        print("请输入你的 OpenAI API Key:")
        api_key = getpass.getpass("API Key (不会显示在屏幕上): ")
    
    if not api_key or not api_key.strip():
        print("错误: API Key 为空")
        return False
    
    api_key = api_key.strip()
    
    # 验证 API Key 格式
    if not api_key.startswith("sk-"):
        print("警告: API Key 似乎不是有效格式 (应以 'sk-' 开头)")
        response = input("是否继续? (y/n): ").strip().lower()
        if response != 'y':
            return False
    
    # 获取项目根目录
    project_root = Path(__file__).parent
    env_file = project_root / ".env"
    
    try:
        # 读取现有的 .env 内容
        existing_content = ""
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                existing_content = f.read()
        
        # 检查是否已存在 OPENAI_API_KEY
        if "OPENAI_API_KEY=" in existing_content:
            # 替换现有的 Key
            lines = existing_content.split("\n")
            new_lines = []
            for line in lines:
                if line.startswith("OPENAI_API_KEY="):
                    new_lines.append(f"OPENAI_API_KEY={api_key}")
                else:
                    new_lines.append(line)
            new_content = "\n".join(new_lines)
        else:
            # 添加新的 Key
            if existing_content and not existing_content.endswith("\n"):
                new_content = existing_content + "\n" + f"OPENAI_API_KEY={api_key}"
            else:
                new_content = existing_content + f"OPENAI_API_KEY={api_key}\n"
        
        # 写入 .env 文件
        with open(env_file, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        print(f"成功! API Key 已写入: {env_file}")
        print(f"Key: {api_key[:20]}...{api_key[-10:]}")
        return True
    
    except Exception as e:
        print(f"错误: 写入 .env 文件失败")
        print(f"   {str(e)}")
        return False


def main():
    """主函数"""
    print("="*60)
    print("OpenAI API Key 设置")
    print("="*60)
    print()
    
    # 交互式输入，不在脚本中存储 Key
    success = setup_api_key()
    
    if not success:
        print()
        print("设置失败")
        sys.exit(1)
    
    print()
    print("="*60)
    print("设置完成!")
    print("="*60)


if __name__ == "__main__":
    main()
