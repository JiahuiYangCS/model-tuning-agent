#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时GPU监控脚本 - 简单版
显示当前GPU利用率、显存使用、功耗等关键指标
"""

import time
from gpu_monitor import PyNVMLMonitor

def main():
    print("=" * 70)
    print("实时GPU监控（每2秒刷新一次，按Ctrl+C停止）")
    print("=" * 70)
    
    monitor = PyNVMLMonitor(gpu_id=0)
    
    try:
        print("\n开始监控...\n")
        while True:
            metrics = monitor.get_instant_metrics()
            if metrics:
                # 清屏效果（简单方式）
                print("\n" + "="*70)
                print(f"⏰ 时间: {time.strftime('%H:%M:%S')}")
                print(f"📊 GPU利用率: {metrics.get('gpu_utilization', 0):.1f}% {'🔥' if metrics.get('gpu_utilization', 0) > 80 else '📈' if metrics.get('gpu_utilization', 0) > 50 else '📉'}")
                print(f"💾 显存使用: {metrics.get('memory_used_mb', 0):.0f}MB / {metrics.get('memory_total_mb', 0):.0f}MB ({metrics.get('memory_utilization', 0):.1f}%)")
                print(f"🌡️  温度: {metrics.get('temperature', 0)}°C")
                print(f"⚡ 功耗: {metrics.get('power_draw', 0):.1f}W")
                print(f"🔧 GPU时钟: {metrics.get('clocks_current', 0)}MHz")
                
                # 状态判断
                util = metrics.get('gpu_utilization', 0)
                if util >= 80:
                    print("\n✅ 状态: GPU高负载运行中！利用率优秀！")
                elif util >= 50:
                    print("\n⚠️  状态: GPU中等负载")
                else:
                    print("\n📉 状态: GPU低负载")
                
                print("="*70)
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print("\n\n监控已停止。")
    except Exception as e:
        print(f"\n错误: {e}")

if __name__ == "__main__":
    main()
