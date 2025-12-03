import os
from openai import OpenAI

# 建议在运行脚本前，在系统环境变量中设置 OPENAI_API_KEY。
# 例如在 Linux / macOS 下：
#   export OPENAI_API_KEY="sk-xxx"
# 在 Windows PowerShell 下：
#   $env:OPENAI_API_KEY="sk-xxx"

client = OpenAI()

print("OpenAI 客户端已初始化 (from openai_client.py)")
