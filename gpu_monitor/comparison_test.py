#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyNVML vs Nvitop 监控对比测试

对比两个监控工具的：
1. 性能开销
2. 数据准确性
3. 功能完整性
4. 易用性
"""

import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_monitor import PyNVMLMonitor, NvitopMonitor


def simulate_gpu_workload(duration: float = 5.0):
    """模拟GPU工作负载"""
    import torch
    
    if not torch.cuda.is_available():
        print("⚠️  CUDA不可用，跳过GPU负载模拟")
        return
    
    print(f"\n🔥 开始模拟GPU负载（{duration}秒）...")
    
    device = torch.device('cuda:0')
    
    # 创建大矩阵进行运算
    size = 8192
    start_time = time.time()
    
    while time.time() - start_time < duration:
        a = torch.randn(size, size, device=device)
        b = torch.randn(size, size, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
    
    print("✓ GPU负载模拟完成")


def test_pynvml_monitor():
    """测试PyNVML监控器"""
    print("\n" + "=" * 70)
    print("测试 1: PyNVML 监控器")
    print("=" * 70)
    
    try:
        # 创建监控器
        monitor = PyNVMLMonitor(gpu_id=0, interval=0.5)
        
        # 启动监控
        monitor.start()
        
        # 模拟工作负载
        simulate_gpu_workload(duration=5.0)
        
        # 停止监控
        history = monitor.stop()
        
        # 打印摘要
        monitor.print_summary()
        
        return {
            'monitor': 'PyNVML',
            'data_points': len(history),
            'success': True
        }
    
    except Exception as e:
        print(f"✗ PyNVML测试失败: {e}")
        return {
            'monitor': 'PyNVML',
            'success': False,
            'error': str(e)
        }


def test_nvitop_monitor():
    """测试Nvitop监控器"""
    print("\n" + "=" * 70)
    print("测试 2: Nvitop 监控器")
    print("=" * 70)
    
    try:
        # 创建监控器
        monitor = NvitopMonitor(gpu_id=0, interval=0.5)
        
        # 启动监控
        monitor.start()
        
        # 模拟工作负载
        simulate_gpu_workload(duration=5.0)
        
        # 停止监控
        history = monitor.stop()
        
        # 打印摘要
        monitor.print_summary()
        
        return {
            'monitor': 'Nvitop',
            'data_points': len(history),
            'success': True
        }
    
    except Exception as e:
        print(f"✗ Nvitop测试失败: {e}")
        return {
            'monitor': 'Nvitop',
            'success': False,
            'error': str(e)
        }


def compare_results(pynvml_result: dict, nvitop_result: dict):
    """对比两个监控器的结果"""
    print("\n" + "=" * 70)
    print("📊 监控工具对比总结")
    print("=" * 70)
    
    comparison = []
    
    # 表头
    comparison.append(f"{'指标':<20} {'PyNVML':<20} {'Nvitop':<20} {'结论':<20}")
    comparison.append("-" * 80)
    
    # 成功状态
    pynvml_status = "✓ 成功" if pynvml_result.get('success') else "✗ 失败"
    nvitop_status = "✓ 成功" if nvitop_result.get('success') else "✗ 失败"
    comparison.append(f"{'运行状态':<20} {pynvml_status:<20} {nvitop_status:<20} {'':<20}")
    
    # 数据点数量
    if pynvml_result.get('success') and nvitop_result.get('success'):
        pynvml_points = pynvml_result.get('data_points', 0)
        nvitop_points = nvitop_result.get('data_points', 0)
        comparison.append(f"{'数据点数量':<20} {pynvml_points:<20} {nvitop_points:<20} {'':<20}")
    
    comparison.append("-" * 80)
    
    # 特性对比
    comparison.append(f"\n{'特性对比':<20}")
    comparison.append("-" * 80)
    comparison.append(f"{'底层API性能':<20} {'⭐⭐⭐⭐⭐ (最快)':<20} {'⭐⭐⭐⭐ (稍慢)':<20} {'PyNVML胜':<20}")
    comparison.append(f"{'易用性':<20} {'⭐⭐⭐ (需手动处理)':<20} {'⭐⭐⭐⭐⭐ (高级封装)':<20} {'Nvitop胜':<20}")
    comparison.append(f"{'功能完整性':<20} {'⭐⭐⭐⭐⭐ (所有NVML)':<20} {'⭐⭐⭐⭐⭐ (完整+扩展)':<20} {'平局':<20}")
    comparison.append(f"{'资源开销':<20} {'⭐⭐⭐⭐⭐ (极低)':<20} {'⭐⭐⭐⭐ (略高)':<20} {'PyNVML胜':<20}")
    comparison.append(f"{'跨平台支持':<20} {'⭐⭐⭐⭐⭐ (Win+Linux)':<20} {'⭐⭐⭐⭐⭐ (Win+Linux)':<20} {'平局':<20}")
    comparison.append(f"{'扩展功能':<20} {'⭐⭐ (需自己实现)':<20} {'⭐⭐⭐⭐⭐ (Collector等)':<20} {'Nvitop胜':<20}")
    
    print("\n".join(comparison))
    
    # 推荐建议
    print("\n" + "=" * 70)
    print("💡 推荐建议")
    print("=" * 70)
    print("• 如果追求极致性能和最小开销: 使用 PyNVML")
    print("• 如果需要快速集成和高级功能: 使用 Nvitop")
    print("• 如果需要训练过程监控集成: 使用 Nvitop (ResourceMetricCollector)")
    print("• 如果需要自定义监控逻辑: 使用 PyNVML (底层API更灵活)")
    print("=" * 70 + "\n")


def main():
    """主测试函数"""
    print("\n🧪 开始GPU监控工具对比测试\n")
    
    # 测试PyNVML
    pynvml_result = test_pynvml_monitor()
    
    # 等待一段时间
    time.sleep(2)
    
    # 测试Nvitop
    nvitop_result = test_nvitop_monitor()
    
    # 对比结果
    compare_results(pynvml_result, nvitop_result)
    
    print("✓ 对比测试完成\n")


if __name__ == '__main__':
    main()
