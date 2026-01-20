#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于nvitop的GPU监控器（简化版，避免Windows兼容性问题）

特点：
- 直接使用Device API
- 线程安全的后台监控
- 不依赖ResourceMetricCollector（Windows上可能有问题）
"""

import time
import threading
from typing import Dict, List, Optional
from collections import defaultdict
from nvitop import Device


class NvitopMonitor:
    """基于nvitop的GPU监控器（简化版）"""
    
    def __init__(self, gpu_id: int = 0, interval: float = 1.0):
        """
        初始化监控器
        
        Args:
            gpu_id: GPU设备ID（CUDA ordinal）
            interval: 监控间隔（秒）
        """
        self.gpu_id = gpu_id
        self.interval = interval
        self.metrics_history = []
        self.monitoring = False
        self._monitor_thread = None
        self._stop_event = threading.Event()
        
        # 初始化设备
        try:
            self.device = Device.cuda(gpu_id)
            self.gpu_name = self.device.name()
            print(f"✓ Nvitop监控器初始化成功: {self.gpu_name}")
        except Exception as e:
            print(f"✗ Nvitop初始化失败: {e}")
            raise
    
    def get_instant_metrics(self) -> Dict:
        """获取即时GPU指标"""
        try:
            # 刷新设备状态
            snapshot = self.device.as_snapshot()
            
            return {
                'timestamp': time.time(),
                'gpu_id': self.gpu_id,
                'gpu_name': self.gpu_name,
                'gpu_utilization': snapshot.gpu_utilization,  # %
                'memory_utilization': snapshot.memory_utilization,  # %
                'memory_used_mb': snapshot.memory_used / (1024 ** 2),  # bytes -> MB
                'memory_total_mb': snapshot.memory_total / (1024 ** 2),
                'memory_free_mb': snapshot.memory_free / (1024 ** 2),
                'memory_percent': snapshot.memory_percent,
                'temperature': snapshot.temperature,  # °C
                'power_usage': snapshot.power_usage / 1000.0,  # mW -> W
                'fan_speed': snapshot.fan_speed,  # %
                'sm_clock_mhz': snapshot.sm_clock,  # MHz
                'mem_clock_mhz': snapshot.memory_clock,  # MHz
            }
        except Exception as e:
            print(f"获取GPU指标失败: {e}")
            return {}
    
    def _monitor_loop(self):
        """监控循环（在后台线程运行）"""
        while not self._stop_event.is_set():
            metrics = self.get_instant_metrics()
            if metrics:
                self.metrics_history.append(metrics)
            time.sleep(self.interval)
    
    def start(self):
        """启动后台监控线程"""
        if self.monitoring:
            print("监控已在运行")
            return
        
        self.metrics_history = []
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        self.monitoring = True
        print(f"✓ Nvitop监控器已启动 (GPU {self.gpu_id}, 间隔 {self.interval}s)")
    
    def stop(self):
        """停止监控"""
        if not self.monitoring:
            return
        
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
        self.monitoring = False
        print(f"✓ Nvitop监控器已停止，共收集 {len(self.metrics_history)} 条数据")
    
    def get_statistics(self) -> Dict:
        """计算统计数据"""
        if not self.metrics_history:
            return {}
        
        stats = defaultdict(list)
        for metrics in self.metrics_history:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and key not in ['timestamp', 'gpu_id']:
                    stats[key].append(value)
        
        result = {}
        for key, values in stats.items():
            if values:
                result[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return result
    
    def print_summary(self):
        """打印监控摘要"""
        if not self.metrics_history:
            print("没有收集到监控数据")
            return
        
        stats = self.get_statistics()
        duration = self.metrics_history[-1]['timestamp'] - self.metrics_history[0]['timestamp']
        
        print("\n" + "=" * 70)
        print("📊 Nvitop 监控摘要")
        print("=" * 70)
        print(f"GPU名称: {self.gpu_name}")
        print(f"数据点数: {len(self.metrics_history)}")
        print(f"监控时长: {duration:.1f}秒")
        print("-" * 70)
        
        if 'gpu_utilization' in stats:
            s = stats['gpu_utilization']
            print(f"GPU利用率:   平均 {s['mean']:.1f}%  最小 {s['min']:.1f}%  最大 {s['max']:.1f}%")
        
        if 'memory_used_mb' in stats:
            s = stats['memory_used_mb']
            total = stats['memory_total_mb']['mean']
            print(f"显存使用:    平均 {s['mean']:.0f}MB  最大 {s['max']:.0f}MB / {total:.0f}MB")
        
        if 'memory_percent' in stats:
            s = stats['memory_percent']
            print(f"显存占用率:  平均 {s['mean']:.1f}%  最大 {s['max']:.1f}%")
        
        if 'temperature' in stats:
            s = stats['temperature']
            print(f"温度:        平均 {s['mean']:.1f}°C  最大 {s['max']:.0f}°C")
        
        if 'power_usage' in stats:
            s = stats['power_usage']
            print(f"功耗:        平均 {s['mean']:.1f}W  最大 {s['max']:.1f}W")
        
        print("=" * 70)
