#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU监控守护进程 - 后台静默记录GPU状态

功能：
1. 后台静默运行，与训练脚本并行
2. 持续记录GPU状态到JSON文件（无实时显示）
3. 训练结束后生成详细的历史分析报告

使用方法：
    # 启动后台监控（在独立终端）
    python gpu_monitor_daemon.py --output gpu_log.json --interval 1.0
    
    # 生成报告
    python gpu_monitor_daemon.py --analyze gpu_log.json --report report.md
"""

import argparse
import json
import time
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict
from gpu_monitor import PyNVMLMonitor


class GPUMonitorDaemon:
    """GPU监控守护进程"""
    
    def __init__(self, output_file: str, interval: float = 1.0, gpu_id: int = 0):
        """
        初始化监控守护进程
        
        Args:
            output_file: 输出JSON文件路径
            interval: 采样间隔（秒）
            gpu_id: GPU设备ID
        """
        self.output_file = output_file
        self.interval = interval
        self.gpu_id = gpu_id
        self.monitor = PyNVMLMonitor(gpu_id=gpu_id)
        self.data_log = []
        self.start_time = None
        
    def start_monitoring(self):
        """启动静默后台监控（无实时显示）"""
        print("=" * 70)
        print("GPU后台监控已启动（静默模式）")
        print("=" * 70)
        print(f"输出文件: {self.output_file}")
        print(f"采样间隔: {self.interval}秒")
        print(f"GPU设备: {self.gpu_id}")
        print("\n监控将在后台静默运行，不会实时显示数据")
        print("按 Ctrl+C 停止监控并保存数据")
        print("=" * 70)
        
        self.start_time = time.time()
        sample_count = 0
        last_print_time = time.time()
        
        try:
            while True:
                # 获取GPU指标
                metrics = self.monitor.get_instant_metrics()
                if metrics:
                    # 添加相对时间和绝对时间
                    metrics['relative_time'] = time.time() - self.start_time
                    metrics['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    self.data_log.append(metrics)
                    sample_count += 1
                    
                    # 每30秒显示一次简要状态（不影响实时性）
                    current_time = time.time()
                    if current_time - last_print_time >= 30:
                        elapsed = current_time - self.start_time
                        print(f"[监控中] 已运行 {elapsed:.0f}秒 | 采样点数: {sample_count} | "
                              f"当前GPU: {metrics.get('gpu_utilization', 0):.1f}% | "
                              f"显存: {metrics.get('memory_used_mb', 0):.0f}MB")
                        last_print_time = current_time
                
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("\n\n监控已停止，正在保存数据...")
            self.save_data()
            print(f"✓ 数据已保存到: {self.output_file}")
            print(f"✓ 共记录 {len(self.data_log)} 条数据")
            print(f"✓ 总监控时长: {time.time() - self.start_time:.1f}秒")
            print(f"\n使用以下命令生成详细报告:")
            print(f"python gpu_monitor_daemon.py --analyze {self.output_file} --report gpu_report.md")
            
    def save_data(self):
        """保存监控数据到JSON文件"""
        data = {
            'metadata': {
                'start_time': datetime.fromtimestamp(self.start_time).strftime('%Y-%m-%d %H:%M:%S'),
                'end_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'duration_seconds': time.time() - self.start_time,
                'total_samples': len(self.data_log),
                'sampling_interval': self.interval,
                'gpu_id': self.gpu_id,
                'gpu_name': self.data_log[0].get('gpu_name', 'Unknown') if self.data_log else 'Unknown'
            },
            'metrics': self.data_log
        }
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class GPUReportGenerator:
    """GPU监控报告生成器"""
    
    def __init__(self, log_file: str):
        """
        初始化报告生成器
        
        Args:
            log_file: 监控日志JSON文件路径
        """
        self.log_file = log_file
        with open(log_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.metadata = self.data['metadata']
        self.metrics = self.data['metrics']
        
    def generate_report(self, output_file: str):
        """
        生成详细的GPU监控分析报告
        
        Args:
            output_file: 输出Markdown报告文件路径
        """
        print(f"正在生成GPU监控报告...")
        print(f"  数据文件: {self.log_file}")
        print(f"  样本数: {len(self.metrics)}")
        print(f"  监控时长: {self.metadata['duration_seconds']:.1f}秒")
        
        report_lines = []
        
        # 标题和元数据
        report_lines.extend(self._generate_header())
        
        # 整体统计摘要
        report_lines.extend(self._generate_overall_summary())
        
        # 时间段分析（按训练轮次分段）
        report_lines.extend(self._generate_time_segment_analysis())
        
        # 详细历史数据表格
        report_lines.extend(self._generate_detailed_history())
        
        # 峰值和异常分析
        report_lines.extend(self._generate_peak_analysis())
        
        # 趋势图数据（可用于可视化）
        report_lines.extend(self._generate_trend_data())
        
        # 写入文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ 报告已生成: {output_file}")
        
    def _generate_header(self) -> List[str]:
        """生成报告头部"""
        return [
            "# GPU监控详细报告",
            "",
            f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**监控开始**: {self.metadata['start_time']}",
            f"**监控结束**: {self.metadata['end_time']}",
            f"**总时长**: {self.metadata['duration_seconds']:.1f}秒 ({self.metadata['duration_seconds']/60:.2f}分钟)",
            f"**GPU设备**: {self.metadata['gpu_name']}",
            f"**采样间隔**: {self.metadata['sampling_interval']}秒",
            f"**数据点数**: {self.metadata['total_samples']}",
            "",
            "---",
            ""
        ]
    
    def _generate_overall_summary(self) -> List[str]:
        """生成整体统计摘要"""
        lines = [
            "## 📊 整体统计摘要",
            ""
        ]
        
        # 计算各指标的统计数据
        stats = self._calculate_statistics([
            'gpu_utilization',
            'memory_used_mb',
            'memory_utilization',
            'temperature',
            'power_draw'
        ])
        
        lines.append("| 指标 | 平均值 | 最小值 | 最大值 | 中位数 |")
        lines.append("|------|--------|--------|--------|--------|")
        
        metrics_info = {
            'gpu_utilization': ('GPU利用率', '%'),
            'memory_used_mb': ('显存使用', 'MB'),
            'memory_utilization': ('显存占用率', '%'),
            'temperature': ('温度', '°C'),
            'power_draw': ('功耗', 'W')
        }
        
        for key, (name, unit) in metrics_info.items():
            if key in stats:
                s = stats[key]
                lines.append(
                    f"| {name} | {s['mean']:.1f}{unit} | "
                    f"{s['min']:.1f}{unit} | {s['max']:.1f}{unit} | "
                    f"{s['median']:.1f}{unit} |"
                )
        
        lines.extend(["", "---", ""])
        return lines
    
    def _generate_time_segment_analysis(self) -> List[str]:
        """生成时间段分析（自适应时间段长度）"""
        total_duration = self.metadata['duration_seconds']
        
        # 自适应时间段长度：总时长的10%，最小30秒，最大120秒
        segment_duration = max(30, min(120, int(total_duration * 0.1)))
        
        lines = [
            "## ⏱️ 时间段详细分析",
            "",
            f"按时间段（每{segment_duration}秒）统计GPU使用情况：",
            f"*注：时间段长度根据总监控时长自适应调整（总时长的10%，范围30-120秒）*",
            ""
        ]
        
        num_segments = int(total_duration / segment_duration) + 1
        
        lines.append("| 时间段 | 时长 | 平均GPU利用率 | 平均显存使用 | 平均温度 | 平均功耗 |")
        lines.append("|--------|------|--------------|--------------|----------|----------|")
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, total_duration)
            
            # 筛选该时间段的数据
            segment_data = [
                m for m in self.metrics
                if start_time <= m['relative_time'] < end_time
            ]
            
            if not segment_data:
                continue
            
            # 计算该段的平均值
            avg_gpu = sum(m.get('gpu_utilization', 0) for m in segment_data) / len(segment_data)
            avg_mem = sum(m.get('memory_used_mb', 0) for m in segment_data) / len(segment_data)
            avg_temp = sum(m.get('temperature', 0) for m in segment_data) / len(segment_data)
            avg_power = sum(m.get('power_draw', 0) for m in segment_data) / len(segment_data)
            
            lines.append(
                f"| {i*60}-{int(end_time)}秒 | {end_time-start_time:.0f}s | "
                f"{avg_gpu:.1f}% | {avg_mem:.0f}MB | "
                f"{avg_temp:.1f}°C | {avg_power:.1f}W |"
            )
        
        lines.extend(["", "---", ""])
        return lines
    
    def _generate_detailed_history(self) -> List[str]:
        """生成详细历史数据表格（每10个样本一行）"""
        lines = [
            "## 📝 详细历史记录",
            "",
            "完整的GPU监控历史数据（每10秒采样点）：",
            ""
        ]
        
        # 每10个样本取一个显示，避免表格过长
        sample_step = max(1, len(self.metrics) // 100)  # 最多显示100行
        
        lines.append("| 时刻 | 相对时间 | GPU利用率 | 显存使用 | 温度 | 功耗 |")
        lines.append("|------|----------|-----------|----------|------|------|")
        
        for i in range(0, len(self.metrics), sample_step):
            m = self.metrics[i]
            lines.append(
                f"| {m.get('datetime', '')} | "
                f"{m.get('relative_time', 0):.1f}s | "
                f"{m.get('gpu_utilization', 0):.1f}% | "
                f"{m.get('memory_used_mb', 0):.0f}MB | "
                f"{m.get('temperature', 0)}°C | "
                f"{m.get('power_draw', 0):.1f}W |"
            )
        
        lines.extend(["", f"*注：表格显示每{sample_step}个采样点，完整数据见JSON文件*", "", "---", ""])
        return lines
    
    def _generate_peak_analysis(self) -> List[str]:
        """生成峰值和异常分析"""
        lines = [
            "## 🔥 峰值与异常分析",
            ""
        ]
        
        # 找出GPU利用率峰值时刻
        max_gpu_metric = max(self.metrics, key=lambda m: m.get('gpu_utilization', 0))
        lines.append(f"**GPU利用率峰值**: {max_gpu_metric.get('gpu_utilization', 0):.1f}%")
        lines.append(f"  - 时刻: {max_gpu_metric.get('datetime', '')}")
        lines.append(f"  - 相对时间: {max_gpu_metric.get('relative_time', 0):.1f}秒")
        lines.append("")
        
        # 找出显存使用峰值时刻
        max_mem_metric = max(self.metrics, key=lambda m: m.get('memory_used_mb', 0))
        lines.append(f"**显存使用峰值**: {max_mem_metric.get('memory_used_mb', 0):.0f}MB ({max_mem_metric.get('memory_utilization', 0):.1f}%)")
        lines.append(f"  - 时刻: {max_mem_metric.get('datetime', '')}")
        lines.append(f"  - 相对时间: {max_mem_metric.get('relative_time', 0):.1f}秒")
        lines.append("")
        
        # 找出温度峰值时刻
        max_temp_metric = max(self.metrics, key=lambda m: m.get('temperature', 0))
        lines.append(f"**温度峰值**: {max_temp_metric.get('temperature', 0)}°C")
        lines.append(f"  - 时刻: {max_temp_metric.get('datetime', '')}")
        lines.append(f"  - 相对时间: {max_temp_metric.get('relative_time', 0):.1f}秒")
        lines.append("")
        
        # 找出功耗峰值时刻
        max_power_metric = max(self.metrics, key=lambda m: m.get('power_draw', 0))
        lines.append(f"**功耗峰值**: {max_power_metric.get('power_draw', 0):.1f}W")
        lines.append(f"  - 时刻: {max_power_metric.get('datetime', '')}")
        lines.append(f"  - 相对时间: {max_power_metric.get('relative_time', 0):.1f}秒")
        lines.append("")
        
        # GPU低利用率时段统计
        low_util_count = sum(1 for m in self.metrics if m.get('gpu_utilization', 0) < 20)
        low_util_percent = (low_util_count / len(self.metrics)) * 100
        lines.append(f"**GPU低利用率（<20%）时段**: {low_util_count}个采样点 ({low_util_percent:.1f}%)")
        lines.append("")
        
        lines.extend(["---", ""])
        return lines
    
    def _generate_trend_data(self) -> List[str]:
        """生成趋势数据（用于可视化）"""
        lines = [
            "## 📈 趋势数据",
            "",
            "GPU利用率随时间变化趋势（前100个采样点）：",
            "",
            "```"
        ]
        
        # 生成简单的ASCII趋势图（前100个点）
        sample_points = self.metrics[:min(100, len(self.metrics))]
        max_util = max(m.get('gpu_utilization', 0) for m in sample_points)
        
        for m in sample_points:
            util = m.get('gpu_utilization', 0)
            bar_length = int((util / 100) * 50)
            lines.append(f"{m.get('relative_time', 0):6.1f}s | {'█' * bar_length} {util:.1f}%")
        
        lines.extend(["```", "", "---", ""])
        return lines
    
    def _calculate_statistics(self, keys: List[str]) -> Dict:
        """计算统计数据"""
        stats = {}
        for key in keys:
            values = [m.get(key, 0) for m in self.metrics if key in m]
            if values:
                values_sorted = sorted(values)
                mid = len(values_sorted) // 2
                # 精确计算中位数（偶数长度取中间两数平均）
                if len(values_sorted) % 2 == 0:
                    median = (values_sorted[mid-1] + values_sorted[mid]) / 2
                else:
                    median = values_sorted[mid]
                
                stats[key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'median': median
                }
        return stats


def main():
    parser = argparse.ArgumentParser(description='GPU监控守护进程')
    parser.add_argument('--output', '-o', type=str, default='gpu_monitor_log.json',
                        help='输出文件路径（JSON格式）')
    parser.add_argument('--interval', '-i', type=float, default=1.0,
                        help='采样间隔（秒）')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU设备ID')
    parser.add_argument('--analyze', '-a', type=str, default=None,
                        help='分析已有的监控日志文件')
    parser.add_argument('--report', '-r', type=str, default='gpu_monitor_report.md',
                        help='生成报告的输出路径')
    
    args = parser.parse_args()
    
    if args.analyze:
        # 分析模式：生成报告
        if not os.path.exists(args.analyze):
            print(f"错误: 文件不存在: {args.analyze}")
            sys.exit(1)
        
        generator = GPUReportGenerator(args.analyze)
        generator.generate_report(args.report)
    else:
        # 监控模式：启动守护进程
        daemon = GPUMonitorDaemon(
            output_file=args.output,
            interval=args.interval,
            gpu_id=args.gpu
        )
        daemon.start_monitoring()


if __name__ == '__main__':
    main()
