#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设置 OpenRouter API Key 到 .env 文件

使用方法:
    python setup_openrouter_api_key.py

行为与 `setup_api_key.py` 类似，会把 `OPENROUTER_API_KEY` 写入项目根目录下的 `.env` 文件（已 .gitignore）。
"""

import os
import sys
import getpass
from pathlib import Path


def setup_openrouter_api_key(api_key=None):
    """将 OpenRouter API Key 写入 .env 文件
    如果没有提供 api_key，将提示用户输入（隐藏输入）
    """

    if not api_key:
        print("请输入你的 OpenRouter API Key:")
        api_key = getpass.getpass("API Key (不会显示在屏幕上): ")

    if not api_key or not api_key.strip():
        print("错误: API Key 为空")
        return False

    api_key = api_key.strip()

    # 获取项目根目录
    project_root = Path(__file__).parent
    env_file = project_root / ".env"

    try:
        # 读取现有的 .env 内容
        existing_content = ""
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                existing_content = f.read()

        # 检查是否已存在 OPENROUTER_API_KEY
        if "OPENROUTER_API_KEY=" in existing_content:
            lines = existing_content.split("\n")
            new_lines = []
            for line in lines:
                if line.startswith("OPENROUTER_API_KEY="):
                    new_lines.append(f"OPENROUTER_API_KEY={api_key}")
                else:
                    new_lines.append(line)
            new_content = "\n".join(new_lines)
        else:
            if existing_content and not existing_content.endswith("\n"):
                new_content = existing_content + "\n" + f"OPENROUTER_API_KEY={api_key}"
            else:
                new_content = existing_content + f"OPENROUTER_API_KEY={api_key}\n"

        with open(env_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"成功! OpenRouter API Key 已写入: {env_file}")
        print(f"Key: {api_key[:20]}...{api_key[-10:]}")
        return True

    except Exception as e:
        print(f"错误: 写入 .env 文件失败")
        print(f"   {str(e)}")
        return False


def main():
    print("=" * 60)
    print("OpenRouter API Key 设置")
    print("=" * 60)
    print()

    success = setup_openrouter_api_key()

    if not success:
        print()
        print("设置失败")
        sys.exit(1)

    print()
    print("=" * 60)
    print("设置完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
