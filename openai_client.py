import os
from openai import OpenAI

# 建议在运行脚本前，在系统环境变量中设置 OPENAI_API_KEY。
# 例如在 Linux / macOS 下：
#   export OPENAI_API_KEY="sk-xxx"
# 在 Windows PowerShell 下：
#   $env:OPENAI_API_KEY="sk-xxx"

_api_key = os.environ.get("OPENAI_API_KEY")
# 如果环境变量未设置，则使用硬编码的 API Key（注意：硬编码密钥存在安全风险）
# 你已要求把 Key 写到脚本中；如果不想硬编码，请继续使用环境变量或更安全的秘密管理方案。
_hardcoded_api_key = "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"

_final_api_key = _api_key if _api_key else _hardcoded_api_key

try:
	client = OpenAI(api_key=_final_api_key)
	print("OpenAI 客户端已初始化 (from openai_client.py)")
except Exception as e:
	client = None
	print("OpenAI 客户端初始化失败（请检查 API Key）：", repr(e))
