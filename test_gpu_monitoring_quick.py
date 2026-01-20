#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU监控和优化快速测试（不包含训练）

测试内容:
1. PyNVML vs Nvitop 监控对比
2. GPU优化器功能演示
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gpu_monitor.comparison_test import main as comparison_test
from gpu_monitor import GPUOptimizer


def test_optimizer():
    """测试GPU优化器"""
    print("\n" + "=" * 70)
    print("测试 2: GPU优化器")
    print("=" * 70)
    
    try:
        optimizer = GPUOptimizer(gpu_id=0)
        print(f"✓ GPU优化器初始化成功: {optimizer.gpu_name} ({optimizer.gpu_memory_total_gb:.1f}GB)")
        
        # 模拟当前配置
        current_config = {
            'batch_size': 8,
            'memory_used_mb': 1024,
            'num_workers': 0,
            'pin_memory': False,
            'use_amp': False
        }
        
        print("\n当前配置:")
        for key, value in current_config.items():
            print(f"  {key}: {value}")
        
        # 检查AMP支持
        if optimizer.can_use_amp():
            print("✓ GPU支持AMP (Compute Capability {:.1f})".format(optimizer.compute_capability))
        
        # 生成优化报告
        print("\n" + "=" * 70)
        print("🔧 GPU优化建议报告")
        print("=" * 70)
        report = optimizer.generate_optimization_report(current_config)
        print(report)
        
        # 应用优化
        print("\n🔧 应用优化...")
        optimized = optimizer.apply_optimizations(current_config)
        
        print("\n优化后配置:")
        for key, value in optimized.items():
            if key in current_config:
                old_value = current_config[key]
                if old_value != value:
                    print(f"  {key}: {old_value} → {value} ✓")
                else:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: None → {value} ✓")
        
        print("\n✓ 优化器测试完成")
        return True
        
    except Exception as e:
        print(f"✗ 优化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 100)
    print(" " * 30 + "GPU监控和优化快速测试")
    print("=" * 100)
    
    success = True
    
    # 测试1: 监控对比
    print("\n阶段 1/2: 监控工具对比")
    print("-" * 100)
    try:
        comparison_test()
    except Exception as e:
        print(f"✗ 监控对比失败: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    # 测试2: GPU优化器
    print("\n阶段 2/2: GPU优化器测试")
    print("-" * 100)
    if not test_optimizer():
        success = False
    
    # 最终总结
    print("\n" + "=" * 100)
    print("📋 测试总结")
    print("=" * 100)
    if success:
        print("✓ 所有测试通过")
        print("\n主要发现:")
        print("1. PyNVML: 最低开销，最快数据采集，适合高性能场景")
        print("2. Nvitop: 功能丰富，易于集成，适合快速开发")
        print("3. GPU优化器: 自动分析GPU状态并提供优化建议")
        print("   - 批量大小自动调整")
        print("   - 混合精度训练(AMP)支持检测")
        print("   - 数据加载器配置优化")
        print("   - PyTorch底层优化设置")
    else:
        print("✗ 部分测试失败")
    print("=" * 100)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
