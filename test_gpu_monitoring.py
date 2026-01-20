#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU监控和优化完整测试

测试流程：
1. 对比PyNVML和Nvitop监控
2. 测试GPU优化器
3. 集成到实际训练中（小规模数据）
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_monitor.comparison_test import main as comparison_test
from gpu_monitor import GPUOptimizer
import torch


def test_gpu_optimizer():
    """测试GPU优化器"""
    print("\n" + "=" * 70)
    print("测试 3: GPU优化器")
    print("=" * 70)
    
    try:
        # 创建优化器
        optimizer = GPUOptimizer(gpu_id=0, target_utilization=0.85)
        
        # 模拟当前配置
        current_config = {
            'batch_size': 8,
            'memory_used_mb': 1024,  # 1GB
            'num_workers': 0,
            'pin_memory': False,
            'use_amp': False,
        }
        
        print("\n当前配置:")
        for key, value in current_config.items():
            print(f"  {key}: {value}")
        
        # 生成优化报告
        report = optimizer.generate_optimization_report(current_config)
        print(report)
        
        # 应用优化
        optimized_config = optimizer.apply_optimizations(current_config)
        
        print("\n优化后配置:")
        for key, value in optimized_config.items():
            if value != current_config.get(key):
                print(f"  {key}: {current_config.get(key)} → {value} ✓")
            else:
                print(f"  {key}: {value}")
        
        return True
    
    except Exception as e:
        print(f"✗ GPU优化器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_integration():
    """测试与训练集成（小规模数据）"""
    print("\n" + "=" * 70)
    print("测试 4: 训练集成测试")
    print("=" * 70)
    
    try:
        from gpu_monitor import NvitopMonitor, GPUOptimizer
        from core.training import train_one_round, make_default_config
        
        # 创建配置（大规模 - 目标GPU利用率80%+）
        config = make_default_config()
        config['QUICK_TEST_MODE'] = False
        config['STSB_TRAIN_SPLIT'] = 'train'  # 全量5749样本
        config['STSB_DEV_SPLIT'] = 'validation'  # 全量1500样本
        config['NUM_TRAIN_EPOCHS'] = 4
        config['TRAIN_BATCH_SIZE'] = 32
        
        print("\n📋 训练配置:")
        print(f"  数据集: STSb")
        print(f"  训练样本: 5749（全量）")
        print(f"  验证样本: 1500（全量）")
        print(f"  Batch Size: {config['TRAIN_BATCH_SIZE']}")
        print(f"  训练轮数: {config['NUM_TRAIN_EPOCHS']}")
        print(f"  预计训练时长: ~2-3分钟")
        print(f"  目标GPU利用率: 80%+")
        
        # 创建监控器和优化器
        monitor = NvitopMonitor(gpu_id=0, interval=1.0)
        optimizer = GPUOptimizer(gpu_id=0)
        
        # 优化配置
        print("\n🔧 应用GPU优化...")
        optimized_config = optimizer.apply_optimizations({
            'batch_size': config['TRAIN_BATCH_SIZE'],
            'memory_used_mb': 3000,  # 估算大batch的显存使用
            'num_workers': 0,
            'pin_memory': False,
            'use_amp': False,
        })
        
        # 更新配置
        if 'batch_size' in optimized_config and optimized_config['batch_size'] > config['TRAIN_BATCH_SIZE']:
            config['TRAIN_BATCH_SIZE'] = min(optimized_config['batch_size'], 16)  # 限制最大值
            print(f"✓ Batch Size 已优化: 8 → {config['TRAIN_BATCH_SIZE']}")
        
        # 启动监控
        print("\n🔍 启动GPU监控...")
        monitor.start()
        
        # 运行训练
        print("\n🚀 开始训练...")
        print("-" * 70)
        summary, metrics = train_one_round(config, round_id=1)
        print("-" * 70)
        
        # 停止监控
        print("\n⏸️  停止监控...")
        monitor.stop()
        monitor.print_summary()
        
        # 训练结果
        print("\n" + "=" * 70)
        print("📊 训练结果")
        print("=" * 70)
        print(f"主评估分数: {summary['main_score']:.4f}")
        print(f"训练时间: {metrics.get('train_runtime', 0):.2f}秒")
        print(f"输出目录: {summary['output_dir']}")
        print(f"GPU设备: {summary.get('device', 'N/A')}")
        print("=" * 70 + "\n")
        
        return True
    
    except Exception as e:
        print(f"✗ 训练集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 100)
    print(" " * 30 + "GPU监控和优化完整测试")
    print("=" * 100 + "\n")
    
    results = {}
    
    # 测试1: 监控工具对比
    print("阶段 1/4: 监控工具对比")
    print("-" * 100)
    try:
        comparison_test()
        results['comparison'] = True
    except Exception as e:
        print(f"✗ 监控对比失败: {e}")
        results['comparison'] = False
    
    # 测试2: GPU优化器
    print("\n阶段 2/4: GPU优化器测试")
    print("-" * 100)
    results['optimizer'] = test_gpu_optimizer()
    
    # 测试3: 训练集成
    print("\n阶段 3/4: 训练集成测试")
    print("-" * 100)
    results['training'] = test_training_integration()
    
    # 总结
    print("\n" + "=" * 100)
    print(" " * 40 + "测试总结")
    print("=" * 100)
    
    for test_name, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name.capitalize()}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！GPU监控和优化模块工作正常。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")
    
    print("=" * 100 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
