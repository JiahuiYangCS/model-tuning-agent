@echo off
REM 设置 PowerShell 编码为 UTF-8 并运行 Python 脚本
chcp 65001 > nul
set PYTHONIOENCODING=utf-8
python run.py
pause
