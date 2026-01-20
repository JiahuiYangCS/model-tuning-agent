#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于pynvml (nvidia-ml-py) 的GPU监控器

特点：
- 官方NVIDIA NVML Python绑定
- 底层API，性能最优
- 最准确的数据
"""

import time
import threading
from typing import Dict, List, Optional
from collections import defaultdict
import pynvml


class PyNVMLMonitor:
    """基于pynvml的GPU监控器"""
    
    def __init__(self, gpu_id: int = 0, interval: float = 1.0):
        """
        初始化监控器
        
        Args:
            gpu_id: GPU设备ID
            interval: 监控间隔（秒）
        """
        self.gpu_id = gpu_id
        self.interval = interval
        self.running = False
        self.metrics_history = []
        self.thread = None
        
        # 初始化NVML
        try:
            pynvml.nvmlInit()
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            self.gpu_name = pynvml.nvmlDeviceGetName(self.handle)
            print(f"✓ PyNVML监控器初始化成功: {self.gpu_name}")
        except Exception as e:
            print(f"✗ PyNVML初始化失败: {e}")
            raise
    
    def get_instant_metrics(self) -> Dict:
        """获取即时GPU指标"""
        try:
            # GPU利用率
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            
            # 内存信息
            mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            
            # 温度
            temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # 功耗
            power = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000.0  # mW -> W
            
            # 风扇转速
            try:
                fan_speed = pynvml.nvmlDeviceGetFanSpeed(self.handle)
            except:
                fan_speed = -1
            
            # 时钟频率
            try:
                sm_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_SM)
                mem_clock = pynvml.nvmlDeviceGetClockInfo(self.handle, pynvml.NVML_CLOCK_MEM)
            except:
                sm_clock = -1
                mem_clock = -1
            
            return {
                'timestamp': time.time(),
                'gpu_id': self.gpu_id,
                'gpu_name': self.gpu_name,
                'gpu_utilization': util.gpu,  # %
                'memory_utilization': util.memory,  # %
                'memory_used_mb': mem.used / (1024 ** 2),  # bytes -> MB
                'memory_total_mb': mem.total / (1024 ** 2),
                'memory_free_mb': mem.free / (1024 ** 2),
                'memory_percent': (mem.used / mem.total) * 100,
                'temperature': temp,  # °C
                'power_usage': power,  # W
                'fan_speed': fan_speed,  # %
                'sm_clock_mhz': sm_clock,  # MHz
                'mem_clock_mhz': mem_clock,  # MHz
            }
        except Exception as e:
            print(f"获取GPU指标失败: {e}")
            return {}
    
    def _monitor_loop(self):
        """监控循环（后台线程）"""
        while self.running:
            metrics = self.get_instant_metrics()
            if metrics:
                self.metrics_history.append(metrics)
            time.sleep(self.interval)
    
    def start(self):
        """启动后台监控"""
        if not self.running:
            self.running = True
            self.metrics_history = []
            self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.thread.start()
            print(f"✓ PyNVML监控器已启动 (GPU {self.gpu_id}, 间隔 {self.interval}s)")
    
    def stop(self) -> List[Dict]:
        """停止监控并返回历史数据"""
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join(timeout=2.0)
            print(f"✓ PyNVML监控器已停止，共收集 {len(self.metrics_history)} 条数据")
        return self.metrics_history
    
    def get_statistics(self) -> Dict:
        """计算统计信息"""
        if not self.metrics_history:
            return {}
        
        stats = defaultdict(list)
        for metrics in self.metrics_history:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value >= 0:
                    stats[key].append(value)
        
        result = {}
        for key, values in stats.items():
            if values:
                result[f'{key}_mean'] = sum(values) / len(values)
                result[f'{key}_min'] = min(values)
                result[f'{key}_max'] = max(values)
        
        return result
    
    def print_summary(self):
        """打印监控摘要"""
        stats = self.get_statistics()
        if not stats:
            print("无监控数据")
            return
        
        print("\n" + "=" * 70)
        print("📊 PyNVML 监控摘要")
        print("=" * 70)
        print(f"GPU名称: {self.gpu_name}")
        print(f"数据点数: {len(self.metrics_history)}")
        print(f"监控时长: {len(self.metrics_history) * self.interval:.1f}秒")
        print("-" * 70)
        print(f"GPU利用率:   平均 {stats.get('gpu_utilization_mean', 0):.1f}%  "
              f"最小 {stats.get('gpu_utilization_min', 0):.1f}%  "
              f"最大 {stats.get('gpu_utilization_max', 0):.1f}%")
        print(f"显存使用:    平均 {stats.get('memory_used_mb_mean', 0):.0f}MB  "
              f"最大 {stats.get('memory_used_mb_max', 0):.0f}MB / "
              f"{stats.get('memory_total_mb_mean', 0):.0f}MB")
        print(f"显存占用率:  平均 {stats.get('memory_percent_mean', 0):.1f}%  "
              f"最大 {stats.get('memory_percent_max', 0):.1f}%")
        print(f"温度:        平均 {stats.get('temperature_mean', 0):.1f}°C  "
              f"最大 {stats.get('temperature_max', 0):.0f}°C")
        print(f"功耗:        平均 {stats.get('power_usage_mean', 0):.1f}W  "
              f"最大 {stats.get('power_usage_max', 0):.1f}W")
        print("=" * 70 + "\n")
    
    def __del__(self):
        """清理资源"""
        try:
            self.stop()
            pynvml.nvmlShutdown()
        except:
            pass
