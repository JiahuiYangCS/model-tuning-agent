#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU利用率优化器

基于nvitop实现，提供：
1. 自动批量大小调整
2. 混合精度训练建议
3. 数据加载优化建议
4. 实时性能监控和建议
"""

import os
import torch
from typing import Dict, Optional, Tuple
from nvitop import Device


class GPUOptimizer:
    """GPU利用率优化器"""
    
    def __init__(self, gpu_id: int = 0, target_utilization: float = 0.85):
        """
        初始化优化器
        
        Args:
            gpu_id: GPU设备ID
            target_utilization: 目标GPU利用率（0.0-1.0）
        """
        self.gpu_id = gpu_id
        self.target_utilization = target_utilization
        
        # 初始化设备
        try:
            self.device = Device.cuda(gpu_id)
            self.gpu_name = self.device.name()
            self.total_memory_gb = self.device.memory_total() / (1024 ** 3)
            print(f"✓ GPU优化器初始化成功: {self.gpu_name} ({self.total_memory_gb:.1f}GB)")
        except Exception as e:
            print(f"✗ GPU优化器初始化失败: {e}")
            raise
        
        # 优化建议
        self.optimizations = {
            'amp_enabled': False,
            'cudnn_benchmark': False,
            'recommended_batch_size': None,
            'num_workers': None,
            'pin_memory': False,
        }
    
    def analyze_current_state(self) -> Dict:
        """分析当前GPU状态"""
        snapshot = self.device.as_snapshot()
        
        gpu_util = snapshot.gpu_utilization
        mem_util = snapshot.memory_percent
        mem_free_gb = snapshot.memory_free / (1024 ** 3)
        
        analysis = {
            'gpu_utilization': gpu_util,
            'memory_utilization': mem_util,
            'memory_free_gb': mem_free_gb,
            'status': 'unknown'
        }
        
        # 分析状态
        if gpu_util < 30:
            if mem_util < 50:
                analysis['status'] = 'underutilized_low_memory'
                analysis['bottleneck'] = 'batch_size_too_small'
            else:
                analysis['status'] = 'underutilized_high_memory'
                analysis['bottleneck'] = 'data_loading_slow'
        elif gpu_util < 60:
            if mem_util < 50:
                analysis['status'] = 'moderate_low_memory'
                analysis['bottleneck'] = 'batch_size_can_increase'
            else:
                analysis['status'] = 'moderate_high_memory'
                analysis['bottleneck'] = 'optimization_needed'
        else:
            analysis['status'] = 'well_utilized'
            analysis['bottleneck'] = 'none'
        
        return analysis
    
    def suggest_batch_size(self, current_batch_size: int, current_memory_mb: float) -> int:
        """
        建议批量大小
        
        Args:
            current_batch_size: 当前batch size
            current_memory_mb: 当前显存使用（MB）
        
        Returns:
            建议的batch size
        """
        mem_free_gb = self.device.memory_free() / (1024 ** 3)
        mem_per_sample = current_memory_mb / current_batch_size
        
        # 保守估计：使用80%的空闲内存
        available_memory_mb = mem_free_gb * 1024 * 0.8
        additional_samples = int(available_memory_mb / mem_per_sample)
        
        suggested_batch_size = current_batch_size + additional_samples
        
        # 限制最大值
        suggested_batch_size = min(suggested_batch_size, current_batch_size * 4)
        suggested_batch_size = max(suggested_batch_size, current_batch_size)
        
        return suggested_batch_size
    
    def enable_amp(self) -> bool:
        """
        检查并启用混合精度训练（AMP）
        
        Returns:
            是否支持AMP
        """
        # 检查CUDA capability
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(self.gpu_id)
            # Tensor Cores需要compute capability >= 7.0 (Volta架构及更新)
            if capability[0] >= 7:
                self.optimizations['amp_enabled'] = True
                print(f"✓ GPU支持AMP (Compute Capability {capability[0]}.{capability[1]})")
                return True
            else:
                print(f"✗ GPU不支持AMP (Compute Capability {capability[0]}.{capability[1]} < 7.0)")
        return False
    
    def optimize_pytorch_settings(self):
        """优化PyTorch设置"""
        # 启用cuDNN auto-tuner
        if not torch.backends.cudnn.benchmark:
            torch.backends.cudnn.benchmark = True
            self.optimizations['cudnn_benchmark'] = True
            print("✓ 已启用 torch.backends.cudnn.benchmark")
        
        # 设置CUDA设备顺序
        if 'CUDA_DEVICE_ORDER' not in os.environ:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            print("✓ 已设置 CUDA_DEVICE_ORDER=PCI_BUS_ID")
    
    def suggest_dataloader_config(self, current_num_workers: int = 0) -> Dict:
        """
        建议DataLoader配置
        
        Args:
            current_num_workers: 当前worker数量
        
        Returns:
            建议的配置
        """
        # 建议worker数量：CPU核心数的1/4到1/2
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        suggested_workers = max(2, min(cpu_count // 2, 8))
        
        config = {
            'num_workers': suggested_workers,
            'pin_memory': True,  # 加速CPU到GPU传输
            'persistent_workers': True,  # 保持worker进程（PyTorch 1.7+）
            'prefetch_factor': 2,  # 每个worker预取的batch数
        }
        
        self.optimizations['num_workers'] = suggested_workers
        self.optimizations['pin_memory'] = True
        
        return config
    
    def generate_optimization_report(self, current_config: Dict) -> str:
        """
        生成优化建议报告
        
        Args:
            current_config: 当前配置
        
        Returns:
            报告字符串
        """
        analysis = self.analyze_current_state()
        
        report = []
        report.append("\n" + "=" * 70)
        report.append("🔧 GPU优化建议报告")
        report.append("=" * 70)
        report.append(f"GPU: {self.gpu_name}")
        report.append(f"当前GPU利用率: {analysis['gpu_utilization']:.1f}%")
        report.append(f"当前显存使用: {analysis['memory_utilization']:.1f}%")
        report.append(f"空闲显存: {analysis['memory_free_gb']:.2f}GB")
        report.append(f"状态: {analysis['status']}")
        report.append(f"瓶颈: {analysis.get('bottleneck', 'unknown')}")
        report.append("-" * 70)
        
        # 批量大小建议
        if 'batch_size' in current_config:
            current_bs = current_config['batch_size']
            current_mem = current_config.get('memory_used_mb', 1000)
            suggested_bs = self.suggest_batch_size(current_bs, current_mem)
            
            if suggested_bs > current_bs:
                report.append(f"📦 批量大小建议:")
                report.append(f"   当前: {current_bs}")
                report.append(f"   建议: {suggested_bs} (提升 {(suggested_bs/current_bs - 1)*100:.1f}%)")
                report.append(f"   原因: 显存还有 {analysis['memory_free_gb']:.2f}GB 空闲")
        
        # AMP建议
        if self.enable_amp() and not current_config.get('use_amp', False):
            report.append(f"⚡ 混合精度训练建议:")
            report.append(f"   建议启用AMP (torch.cuda.amp)")
            report.append(f"   预期效果: 训练速度提升1.5-2.5x，显存占用减少40-50%")
        
        # DataLoader建议
        dataloader_config = self.suggest_dataloader_config(current_config.get('num_workers', 0))
        current_workers = current_config.get('num_workers', 0)
        if dataloader_config['num_workers'] > current_workers:
            report.append(f"📥 数据加载优化建议:")
            report.append(f"   num_workers: {current_workers} → {dataloader_config['num_workers']}")
            report.append(f"   pin_memory: {current_config.get('pin_memory', False)} → True")
            report.append(f"   persistent_workers: False → True")
            report.append(f"   预期效果: 减少GPU等待数据的时间")
        
        # PyTorch设置
        if not self.optimizations['cudnn_benchmark']:
            report.append(f"⚙️  PyTorch优化建议:")
            report.append(f"   启用 torch.backends.cudnn.benchmark = True")
            report.append(f"   预期效果: 卷积操作加速5-15%")
        
        report.append("=" * 70 + "\n")
        
        return "\n".join(report)
    
    def apply_optimizations(self, config: Dict) -> Dict:
        """
        应用优化到配置
        
        Args:
            config: 当前配置
        
        Returns:
            优化后的配置
        """
        optimized_config = config.copy()
        
        # 启用AMP
        if self.enable_amp():
            optimized_config['use_amp'] = True
        
        # 优化PyTorch设置
        self.optimize_pytorch_settings()
        
        # 调整批量大小
        if 'batch_size' in config and 'memory_used_mb' in config:
            suggested_bs = self.suggest_batch_size(
                config['batch_size'],
                config['memory_used_mb']
            )
            if suggested_bs > config['batch_size']:
                optimized_config['batch_size'] = suggested_bs
        
        # 优化DataLoader
        dataloader_config = self.suggest_dataloader_config(config.get('num_workers', 0))
        optimized_config.update(dataloader_config)
        
        return optimized_config
