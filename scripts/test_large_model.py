#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试大模型配置

快速测试新的大模型配置是否能正常运行，并显示GPU利用率
"""

import sys
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from sentence_transformers import SentenceTransformer
from config import DEFAULT_CONFIG
from gpu_monitor.pynvml_monitor import PyNVMLMonitor

def test_model_config():
    """测试模型配置"""
    print("=" * 70)
    print("🧪 测试大模型配置")
    print("=" * 70)
    
    # 检查GPU
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到CUDA")
        return
    
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA版本: {torch.version.cuda}")
    
    # 初始化GPU监控
    monitor = PyNVMLMonitor()
    
    # 显示配置
    print(f"\n📋 当前配置:")
    print(f"   模型: {DEFAULT_CONFIG['BASE_MODEL']}")
    print(f"   训练Batch Size: {DEFAULT_CONFIG['TRAIN_BATCH_SIZE']}")
    print(f"   评估Batch Size: {DEFAULT_CONFIG['EVAL_BATCH_SIZE']}")
    print(f"   训练数据: {DEFAULT_CONFIG['STSB_TRAIN_SPLIT']}")
    
    # 获取初始GPU状态
    initial_stats = monitor.get_instant_metrics()
    print(f"\n📊 初始GPU状态:")
    print(f"   GPU利用率: {initial_stats.get('gpu_utilization', 0):.1f}%")
    print(f"   显存使用: {initial_stats.get('memory_used_mb', 0):.0f}MB")
    
    # 加载模型
    print(f"\n🔧 加载模型...")
    try:
        model = SentenceTransformer(DEFAULT_CONFIG['BASE_MODEL'])
        model.to('cuda')
        print("✅ 模型加载成功")
        
        # 检查模型大小
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📊 模型信息:")
        print(f"   总参数量: {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"   可训练参数: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        
        # 加载后GPU状态
        after_load_stats = monitor.get_instant_metrics()
        print(f"\n📊 加载模型后GPU状态:")
        print(f"   GPU利用率: {after_load_stats.get('gpu_utilization', 0):.1f}%")
        print(f"   显存使用: {after_load_stats.get('memory_used_mb', 0):.0f}MB")
        print(f"   显存增加: {after_load_stats.get('memory_used_mb', 0) - initial_stats.get('memory_used_mb', 0):.0f}MB")
        
        # 测试推理
        print(f"\n🧪 测试推理...")
        test_sentences = [
            "This is a test sentence for model inference.",
            "Another test sentence to check the model.",
        ] * (DEFAULT_CONFIG['TRAIN_BATCH_SIZE'] // 2)  # 创建一个batch
        
        import time
        start_time = time.time()
        embeddings = model.encode(test_sentences, batch_size=DEFAULT_CONFIG['EVAL_BATCH_SIZE'], show_progress_bar=False)
        inference_time = time.time() - start_time
        
        print(f"✅ 推理完成")
        print(f"   处理样本数: {len(test_sentences)}")
        print(f"   推理时间: {inference_time:.2f}秒")
        print(f"   吞吐量: {len(test_sentences)/inference_time:.1f} samples/sec")
        
        # 推理后GPU状态
        after_inference_stats = monitor.get_instant_metrics()
        print(f"\n📊 推理后GPU状态:")
        print(f"   GPU利用率: {after_inference_stats.get('gpu_utilization', 0):.1f}%")
        print(f"   显存使用: {after_inference_stats.get('memory_used_mb', 0):.0f}MB")
        print(f"   峰值显存占比: {(after_inference_stats.get('memory_used_mb', 0) / after_inference_stats.get('memory_total_mb', 1)) * 100:.1f}%")
        
        # 估算训练时的显存占用
        estimated_training_memory = after_inference_stats.get('memory_used_mb', 0) * 2.5  # 训练通常是推理的2-3倍
        memory_safe = estimated_training_memory < after_inference_stats.get('memory_total_mb', 1) * 0.9
        
        print(f"\n💡 训练显存估算:")
        print(f"   预估训练显存: {estimated_training_memory:.0f}MB")
        print(f"   显存总量: {after_inference_stats.get('memory_total_mb', 0):.0f}MB")
        print(f"   预估占比: {(estimated_training_memory / after_inference_stats.get('memory_total_mb', 1)) * 100:.1f}%")
        
        if memory_safe:
            print(f"   ✅ 显存充足，可以安全训练")
        else:
            print(f"   ⚠️  显存可能不足，建议降低batch size到 {DEFAULT_CONFIG['TRAIN_BATCH_SIZE'] // 2}")
        
        # 清理
        del model
        torch.cuda.empty_cache()
        
        print(f"\n{'='*70}")
        print("✅ 测试完成")
        print("="*70)
        
        # 总结建议
        print(f"\n📌 配置建议:")
        print(f"   当前配置: BASE_MODEL = '{DEFAULT_CONFIG['BASE_MODEL']}'")
        print(f"   当前Batch Size: {DEFAULT_CONFIG['TRAIN_BATCH_SIZE']}")
        
        if memory_safe:
            print(f"   💡 建议: 配置合理，可以开始训练")
            print(f"   预期GPU利用率: 60-80% (比小模型高约20-30%)")
        else:
            print(f"   ⚠️  建议: 降低batch size或使用梯度累积")
        
        print(f"\n🚀 开始训练:")
        print(f"   python run.py")
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n❌ 显存不足!")
            print(f"   建议:")
            print(f"   1. 降低batch size: TRAIN_BATCH_SIZE = {DEFAULT_CONFIG['TRAIN_BATCH_SIZE'] // 2}")
            print(f"   2. 或使用梯度累积: GRAD_ACC_STEPS = 2")
        else:
            print(f"\n❌ 错误: {e}")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_config()
