#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 Windows 终端中文乱码问题

在 Windows PowerShell 中运行 Python 脚本时，如果出现中文乱码，
可以先运行这个脚本来设置正确的编码。

用法:
    python fix_encoding.py
"""
import sys
import os
import locale

def fix_windows_console_encoding():
    """修复 Windows 控制台编码"""
    if sys.platform == 'win32':
        # 设置环境变量
        os.environ['PYTHONIOENCODING'] = 'utf-8'
        
        # 尝试设置控制台代码页为 UTF-8 (65001)
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # 设置控制台输出代码页
            kernel32.SetConsoleOutputCP(65001)
            # 设置控制台输入代码页
            kernel32.SetConsoleCP(65001)
            print("✓ 已将控制台代码页设置为 UTF-8 (CP65001)")
        except Exception as e:
            print(f"⚠️  设置控制台代码页失败: {e}")
        
        # 重新配置 stdout 和 stderr
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
            print("✓ 已重新配置 stdout/stderr 编码为 UTF-8")
        except AttributeError:
            # Python < 3.7 不支持 reconfigure
            try:
                import codecs
                sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
                sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
                print("✓ 已使用 codecs 重新配置 stdout/stderr 编码为 UTF-8")
            except Exception as e:
                print(f"⚠️  重新配置编码失败: {e}")
        
        # 显示当前编码信息
        print("\n当前编码信息:")
        print(f"  系统默认编码: {sys.getdefaultencoding()}")
        print(f"  文件系统编码: {sys.getfilesystemencoding()}")
        print(f"  stdout 编码: {sys.stdout.encoding}")
        print(f"  stderr 编码: {sys.stderr.encoding}")
        try:
            print(f"  locale 编码: {locale.getpreferredencoding()}")
        except:
            pass
        
        print("\n测试中文显示:")
        print("  你好，世界！")
        print("  STSb Auto-Tune Agent (自动调参)")
        print("  ✓ 如果能正常显示以上中文，说明编码已修复")
        
    else:
        print("✓ 非 Windows 系统，通常不需要特殊编码设置")

if __name__ == "__main__":
    print("=" * 70)
    print("Windows 终端中文编码修复工具")
    print("=" * 70)
    print()
    fix_windows_console_encoding()
    print()
    print("=" * 70)
    print("修复完成！现在可以运行其他脚本了")
    print("=" * 70)
