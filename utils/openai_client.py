"""
OpenAI Client module / OpenAI 客户端模块

初始化和管理 OpenAI API 客户端
"""

import os
from pathlib import Path
from openai import OpenAI


def _load_api_key():
    """
    按优先级加载 API Key
    Priority: .env file > environment variable > None
    """
    # 方式 1: 从 .env 文件读取（推荐）
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.split("=", 1)[1].strip()
                        if key:
                            return key
        except Exception as e:
            print(f"⚠️  警告 / Warning: 读取 .env 文件失败: {e}")
    
    # 方式 2: 从系统环境变量读取
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    
    # 都没有则返回 None
    return None


_final_api_key = _load_api_key()

try:
    client = OpenAI(api_key=_final_api_key)
    print("✅ OpenAI 客户端已初始化 (OpenAI API ready)")
except Exception as e:
    client = None
    print(f"❌ OpenAI 客户端初始化失败 / OpenAI client initialization failed:")
    print(f"   错误 / Error: {repr(e)}")
    print(f"   请运行 / Please run: python setup_api_key.py")
