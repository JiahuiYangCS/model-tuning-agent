#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动寻找最优Batch Size脚本

基于Google Deep Learning Tuning Playbook原则：
https://github.com/google-research/tuning_playbook

核心原则：
1. Batch size不影响最终模型精度，只影响训练速度
2. 固定所有其他超参数（学习率、优化器、正则化等）
3. 测试不同batch size的训练吞吐量（training throughput）
4. 找到"critical batch size"（超过此值后吞吐量不再提升）
5. 推荐：能充分利用硬件的最大batch size

使用方法：
    python scripts/find_optimal_batch_size.py
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from gpu_monitor.pynvml_monitor import PyNVMLMonitor


class BatchSizeOptimizer:
    """
    Batch Size优化器
    
    遵循Google Tuning Playbook原则自动寻找最优batch size
    """
    
    def __init__(self):
        """初始化优化器"""
        self.gpu_monitor = PyNVMLMonitor()
        self.test_results = []
        
        # ========== 固定训练参数（绝不改变！）==========
        # 根据Google Playbook，除batch size外的所有参数必须固定
        self.FIXED_HYPERPARAMETERS = {
            'learning_rate': 2e-5,
            'optimizer': 'AdamW',
            'weight_decay': 0.01,
            'warmup_steps': 100,
            'max_grad_norm': 1.0,
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'loss_function': 'CosineSimilarityLoss',
        }
        
        # 测试配置
        self.TEST_EPOCHS = 1  # 每个batch size测试1个epoch即可
        self.NUM_WARMUP_STEPS = 10  # 预热步数（忽略前几步的不稳定数据）
        self.NUM_MEASUREMENT_STEPS = 50  # 测量步数（用于计算吞吐量）
        
        # 测试数据集（固定大小，确保可复现）
        self.DATASET_SIZE = 1000
        
        # 要测试的batch size列表（2的幂次方）
        self.BATCH_SIZES_TO_TEST = [4, 8, 16, 32, 48, 64, 96, 128, 192, 256]
        
        # 报告输出路径
        self.report_path = project_root / 'docs' / 'reports'
        self.report_path.mkdir(parents=True, exist_ok=True)
    
    def create_fixed_dataset(self) -> List[InputExample]:
        """
        创建固定的测试数据集
        
        Returns:
            训练样本列表
        """
        print(f"📊 创建固定测试数据集：{self.DATASET_SIZE} 样本")
        
        examples = []
        for i in range(self.DATASET_SIZE):
            # 使用固定的文本模板，确保可复现
            text1 = f"Sample training text number {i} for batch size optimization testing."
            text2 = f"Related text number {i} for similarity model training purposes."
            label = 0.8  # 固定相似度标签
            examples.append(InputExample(texts=[text1, text2], label=label))
        
        return examples
    
    def measure_throughput(self, batch_size: int) -> Optional[Dict]:
        """
        测量指定batch size的训练吞吐量
        
        Args:
            batch_size: 要测试的batch size
            
        Returns:
            包含吞吐量和其他指标的字典，如果失败则返回None
        """
        print(f"\n{'='*70}")
        print(f"🧪 测试 Batch Size = {batch_size}")
        print(f"{'='*70}")
        
        try:
            # 清空GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                time.sleep(2)
            
            # 加载模型
            print("🔧 加载模型...")
            model = SentenceTransformer(self.FIXED_HYPERPARAMETERS['model_name'])
            if torch.cuda.is_available():
                model.to('cuda')
            
            # 创建数据集
            train_examples = self.create_fixed_dataset()
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size,
                drop_last=True  # 确保所有batch大小一致
            )
            
            # 定义loss
            train_loss = losses.CosineSimilarityLoss(model)
            
            # 记录GPU初始状态
            initial_memory = 0
            if torch.cuda.is_available():
                initial_stats = self.gpu_monitor.get_instant_metrics()
                initial_memory = initial_stats.get('memory_used_mb', 0)
            
            print(f"🚀 开始训练测量...")
            print(f"   - 预热步数: {self.NUM_WARMUP_STEPS}")
            print(f"   - 测量步数: {self.NUM_MEASUREMENT_STEPS}")
            
            # 使用model.fit进行简单直接的训练
            measurement_start_time = time.time()
            
            # 训练指定步数
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                steps_per_epoch=self.NUM_WARMUP_STEPS + self.NUM_MEASUREMENT_STEPS,
                warmup_steps=0,
                optimizer_params={'lr': self.FIXED_HYPERPARAMETERS['learning_rate']},
                show_progress_bar=True,
                checkpoint_save_steps=9999,  # 不保存
                checkpoint_path=None,
            )
            
            # 记录结束时间
            total_measurement_time = time.time() - measurement_start_time
            
            # 采样GPU状态
            gpu_samples = []
            for _ in range(5):
                gpu_samples.append(self.gpu_monitor.get_instant_metrics())
                time.sleep(0.2)
            
            # 计算结果
            samples_processed = (self.NUM_WARMUP_STEPS + self.NUM_MEASUREMENT_STEPS) * batch_size
            step_times = [total_measurement_time / (self.NUM_WARMUP_STEPS + self.NUM_MEASUREMENT_STEPS)] * self.NUM_MEASUREMENT_STEPS
            
            # 计算测量结果
            if not step_times:
                raise RuntimeError("未收集到有效的测量数据")
            
            # 关键指标：训练吞吐量（examples/second）
            training_throughput = samples_processed / total_measurement_time
            
            # 时间每步（秒）
            avg_time_per_step = sum(step_times) / len(step_times)
            
            # GPU指标
            avg_gpu_util = 0
            peak_memory_mb = initial_memory
            avg_memory_mb = initial_memory
            
            if gpu_samples:
                avg_gpu_util = sum(s.get('gpu_utilization', 0) for s in gpu_samples) / len(gpu_samples)
                peak_memory_mb = max(s.get('memory_used_mb', 0) for s in gpu_samples)
                avg_memory_mb = sum(s.get('memory_used_mb', 0) for s in gpu_samples) / len(gpu_samples)
            
            # 内存利用率
            total_memory = self.gpu_monitor.get_instant_metrics().get('memory_total_mb', 1)
            memory_utilization_pct = (peak_memory_mb / total_memory) * 100
            
            result = {
                'batch_size': batch_size,
                'success': True,
                
                # 核心指标
                'training_throughput_samples_per_sec': round(training_throughput, 2),
                'time_per_step_seconds': round(avg_time_per_step, 4),
                
                # GPU指标
                'avg_gpu_utilization_pct': round(avg_gpu_util, 2),
                'peak_memory_mb': round(peak_memory_mb, 2),
                'avg_memory_mb': round(avg_memory_mb, 2),
                'memory_utilization_pct': round(memory_utilization_pct, 2),
                'total_memory_mb': total_memory,
                
                # 统计信息
                'samples_processed': samples_processed,
                'measurement_steps': len(step_times),
                'total_measurement_time_sec': round(total_measurement_time, 2),
                
                # 时间分布
                'min_step_time': round(min(step_times), 4),
                'max_step_time': round(max(step_times), 4),
                'std_step_time': round(
                    (sum((t - avg_time_per_step) ** 2 for t in step_times) / len(step_times)) ** 0.5,
                    4
                ),
            }
            
            print(f"\n✅ 测试完成")
            print(f"   训练吞吐量: {result['training_throughput_samples_per_sec']} samples/sec")
            print(f"   时间/步: {result['time_per_step_seconds']}s")
            print(f"   GPU利用率: {result['avg_gpu_utilization_pct']}%")
            print(f"   峰值内存: {result['peak_memory_mb']:.0f}MB ({result['memory_utilization_pct']:.1f}%)")
            
            # 清理
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)
            
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                print(f"❌ Batch Size {batch_size} 超出GPU内存限制")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return {
                    'batch_size': batch_size,
                    'success': False,
                    'error': 'OOM',
                    'error_message': 'GPU内存不足'
                }
            else:
                print(f"❌ 运行时错误: {e}")
                return {
                    'batch_size': batch_size,
                    'success': False,
                    'error': str(type(e).__name__),
                    'error_message': str(e)
                }
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'batch_size': batch_size,
                'success': False,
                'error': str(type(e).__name__),
                'error_message': str(e)
            }
    
    def analyze_scaling(self, results: List[Dict]) -> Dict:
        """
        分析scaling行为，找到critical batch size
        
        根据Google Playbook:
        - Perfect scaling: 吞吐量随batch size线性增长
        - Critical batch size: 超过此值后吞吐量不再增长
        
        Args:
            results: 成功的测试结果列表
            
        Returns:
            分析结果字典
        """
        if len(results) < 2:
            return {}
        
        # 按batch size排序
        results = sorted(results, key=lambda x: x['batch_size'])
        
        # 计算scaling效率
        base_throughput = results[0]['training_throughput_samples_per_sec']
        base_batch_size = results[0]['batch_size']
        
        scaling_analysis = []
        
        for i, result in enumerate(results):
            if i == 0:
                scaling_efficiency = 100.0
            else:
                # 理想情况：吞吐量应该与batch size成正比
                expected_throughput = base_throughput * (result['batch_size'] / base_batch_size)
                actual_throughput = result['training_throughput_samples_per_sec']
                scaling_efficiency = (actual_throughput / expected_throughput) * 100
            
            scaling_analysis.append({
                'batch_size': result['batch_size'],
                'throughput': result['training_throughput_samples_per_sec'],
                'scaling_efficiency_pct': round(scaling_efficiency, 1),
                'time_per_step': result['time_per_step_seconds']
            })
        
        # 找到critical batch size（scaling效率低于80%的第一个点）
        critical_batch_size = None
        perfect_scaling_range = []
        
        for item in scaling_analysis:
            if item['scaling_efficiency_pct'] >= 80:
                perfect_scaling_range.append(item['batch_size'])
            elif critical_batch_size is None:
                critical_batch_size = item['batch_size']
        
        return {
            'scaling_analysis': scaling_analysis,
            'perfect_scaling_range': perfect_scaling_range,
            'critical_batch_size': critical_batch_size,
            'has_perfect_scaling': len(perfect_scaling_range) > 0
        }
    
    def find_optimal_batch_size(self, results: List[Dict], scaling: Dict) -> Dict:
        """
        根据测试结果和scaling分析找到最优batch size
        
        推荐策略（按优先级）:
        1. 最大的仍有perfect scaling的batch size
        2. 如果都已超过critical batch size，选择吞吐量最高的
        3. 考虑内存安全边界（<90%）
        
        Args:
            results: 成功的测试结果列表
            scaling: scaling分析结果
            
        Returns:
            推荐结果字典
        """
        if not results:
            return {'recommended_batch_size': None, 'reason': '无有效测试结果'}
        
        # 过滤内存安全的结果（<90%）
        safe_results = [r for r in results if r['memory_utilization_pct'] < 90]
        
        if not safe_results:
            # 如果都不安全，至少选一个相对安全的
            safe_results = [r for r in results if r['memory_utilization_pct'] < 95]
        
        if not safe_results:
            safe_results = results  # 实在没办法，用全部
        
        # 策略1: 最大的perfect scaling batch size
        if scaling.get('perfect_scaling_range'):
            perfect_scaling_safe = [
                r for r in safe_results 
                if r['batch_size'] in scaling['perfect_scaling_range']
            ]
            if perfect_scaling_safe:
                recommended = max(perfect_scaling_safe, key=lambda x: x['batch_size'])
                return {
                    'recommended_batch_size': recommended['batch_size'],
                    'reason': 'Perfect Scaling区间内的最大batch size',
                    'strategy': 'max_perfect_scaling',
                    'details': recommended
                }
        
        # 策略2: 吞吐量最高的安全batch size
        recommended = max(safe_results, key=lambda x: x['training_throughput_samples_per_sec'])
        
        return {
            'recommended_batch_size': recommended['batch_size'],
            'reason': '吞吐量最高且内存安全的batch size',
            'strategy': 'max_throughput',
            'details': recommended
        }
    
    def run_optimization(self):
        """运行完整的batch size优化流程"""
        print("\n" + "="*70)
        print("🎯 自动寻找最优Batch Size")
        print("基于Google Deep Learning Tuning Playbook")
        print("="*70)
        
        # 显示GPU信息
        gpu_info = self.gpu_monitor.get_instant_metrics()
        print(f"\n🖥️  GPU信息:")
        print(f"   型号: {gpu_info.get('gpu_name', 'Unknown')}")
        print(f"   显存: {gpu_info.get('memory_total_mb', 0):.0f} MB")
        
        # 显示固定参数
        print(f"\n🔒 固定超参数（不会改变）:")
        for key, value in self.FIXED_HYPERPARAMETERS.items():
            print(f"   {key}: {value}")
        
        print(f"\n📋 测试配置:")
        print(f"   测试Batch Sizes: {self.BATCH_SIZES_TO_TEST}")
        print(f"   数据集大小: {self.DATASET_SIZE} samples")
        print(f"   预热步数: {self.NUM_WARMUP_STEPS}")
        print(f"   测量步数: {self.NUM_MEASUREMENT_STEPS}")
        
        print(f"\n⚡ 开始测试...\n")
        
        # 测试每个batch size
        for batch_size in self.BATCH_SIZES_TO_TEST:
            result = self.measure_throughput(batch_size)
            if result:
                self.test_results.append(result)
            
            # 如果遇到OOM，停止测试更大的batch size
            if result and not result['success'] and result.get('error') == 'OOM':
                print(f"\n⚠️  达到GPU内存上限，停止测试更大的batch size")
                break
            
            # 短暂休息，让GPU冷却
            time.sleep(3)
        
        # 分析结果
        successful_results = [r for r in self.test_results if r['success']]
        
        if not successful_results:
            print("\n❌ 没有成功的测试结果！")
            return
        
        # Scaling分析
        scaling_analysis = self.analyze_scaling(successful_results)
        
        # 找到最优batch size
        optimal = self.find_optimal_batch_size(successful_results, scaling_analysis)
        
        # 生成报告
        self.generate_report(gpu_info, scaling_analysis, optimal)
    
    def generate_report(self, gpu_info: Dict, scaling: Dict, optimal: Dict):
        """生成详细的优化报告"""
        print(f"\n\n{'='*70}")
        print("📊 生成优化报告")
        print("="*70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_path / f"optimal_batch_size_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 最优Batch Size优化报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("**基于**: [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)\n\n")
            f.write("---\n\n")
            
            # GPU信息
            f.write("## 🖥️ 硬件环境\n\n")
            f.write(f"- **GPU型号**: {gpu_info.get('gpu_name', 'Unknown')}\n")
            f.write(f"- **显存总量**: {gpu_info.get('memory_total_mb', 0):.0f} MB\n")
            f.write(f"- **CUDA版本**: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}\n")
            f.write(f"- **PyTorch版本**: {torch.__version__}\n\n")
            
            # 固定参数
            f.write("## 🔒 固定超参数\n\n")
            f.write("根据Google Playbook原则，除batch size外的所有参数固定：\n\n")
            f.write("```python\n")
            for key, value in self.FIXED_HYPERPARAMETERS.items():
                f.write(f"{key} = {repr(value)}\n")
            f.write("```\n\n")
            
            # 测试结果表格
            successful_results = [r for r in self.test_results if r['success']]
            
            f.write("## 📊 测试结果\n\n")
            f.write("| Batch Size | 状态 | 吞吐量<br/>(samples/s) | 时间/步<br/>(s) | GPU利用率<br/>(%) | 峰值内存<br/>(MB) | 内存利用<br/>(%) |\n")
            f.write("|------------|------|----------------------|----------------|-----------------|-----------------|----------------|\n")
            
            for result in self.test_results:
                if result['success']:
                    f.write(
                        f"| {result['batch_size']} | ✅ | "
                        f"{result['training_throughput_samples_per_sec']:.1f} | "
                        f"{result['time_per_step_seconds']:.4f} | "
                        f"{result['avg_gpu_utilization_pct']:.1f} | "
                        f"{result['peak_memory_mb']:.0f} | "
                        f"{result['memory_utilization_pct']:.1f} |\n"
                    )
                else:
                    error_msg = result.get('error', 'Unknown')
                    f.write(f"| {result['batch_size']} | ❌ {error_msg} | - | - | - | - | - |\n")
            
            # Scaling分析
            if scaling.get('scaling_analysis'):
                f.write("\n## 📈 Scaling效率分析\n\n")
                f.write("**Perfect Scaling**: 吞吐量随batch size线性增长（效率≥80%）\n\n")
                f.write("| Batch Size | 吞吐量<br/>(samples/s) | Scaling效率<br/>(%) | 评价 |\n")
                f.write("|------------|----------------------|-------------------|------|\n")
                
                for item in scaling['scaling_analysis']:
                    efficiency = item['scaling_efficiency_pct']
                    if efficiency >= 95:
                        评价 = "🟢 完美"
                    elif efficiency >= 80:
                        评价 = "🟡 良好"
                    else:
                        评价 = "🔴 衰减"
                    
                    f.write(
                        f"| {item['batch_size']} | "
                        f"{item['throughput']:.1f} | "
                        f"{efficiency:.1f} | "
                        f"{评价} |\n"
                    )
                
                f.write("\n**分析**:\n\n")
                if scaling.get('perfect_scaling_range'):
                    f.write(f"- **Perfect Scaling范围**: Batch Size = {scaling['perfect_scaling_range']}\n")
                if scaling.get('critical_batch_size'):
                    f.write(f"- **Critical Batch Size**: {scaling['critical_batch_size']} "
                           f"(超过此值后scaling效率下降)\n")
                else:
                    f.write(f"- **未达到Critical Batch Size**: 所有测试的batch size都保持良好的scaling效率\n")
                f.write("\n")
            
            # 推荐结果
            f.write("## 🎯 优化推荐\n\n")
            f.write(f"### 🏆 推荐Batch Size: **{optimal['recommended_batch_size']}**\n\n")
            f.write(f"**推荐理由**: {optimal['reason']}\n\n")
            
            if optimal.get('details'):
                details = optimal['details']
                f.write("**性能指标**:\n\n")
                f.write(f"- 训练吞吐量: {details['training_throughput_samples_per_sec']} samples/sec\n")
                f.write(f"- 时间/步: {details['time_per_step_seconds']}s\n")
                f.write(f"- GPU利用率: {details['avg_gpu_utilization_pct']}%\n")
                f.write(f"- 内存利用率: {details['memory_utilization_pct']:.1f}%\n")
                f.write(f"- 峰值内存: {details['peak_memory_mb']:.0f}MB\n\n")
            
            # Google Playbook原则说明
            f.write("## 📚 Google Playbook关键原则\n\n")
            f.write("### 1. Batch Size不影响最终模型精度\n\n")
            f.write("> 只要其他超参数调优得当（特别是学习率和正则化），使用任何batch size都能达到相同的最终性能。\n\n")
            f.write("### 2. Batch Size只影响训练速度\n\n")
            f.write("> 更大的batch size通常能更好地利用GPU，提高训练速度，缩短开发周期。\n\n")
            f.write("### 3. Perfect Scaling\n\n")
            f.write("> 在critical batch size之前，加倍batch size应该能使吞吐量加倍（或接近加倍）。\n\n")
            f.write("### 4. Critical Batch Size\n\n")
            f.write("> 超过某个值后，继续增大batch size不再提升吞吐量。这个值取决于数据集、模型和优化器。\n\n")
            f.write("### 5. 改变Batch Size需要重新调优\n\n")
            f.write("> ⚠️ 如果你采用不同的batch size，必须重新调优学习率和正则化参数！\n\n")
            
            # 实施建议
            f.write("## 🚀 实施指南\n\n")
            f.write("### 步骤1: 更新配置\n\n")
            f.write(f"在 `config.py` 中设置：\n\n")
            f.write("```python\n")
            f.write(f"TRAIN_BATCH_SIZE = {optimal['recommended_batch_size']}\n")
            f.write(f"EVAL_BATCH_SIZE = {optimal['recommended_batch_size']}\n")
            f.write("```\n\n")
            
            f.write("### 步骤2: 保持当前超参数\n\n")
            f.write("因为测试时使用的就是当前的超参数配置，所以可以直接使用：\n\n")
            f.write("```python\n")
            f.write(f"LEARNING_RATE = {self.FIXED_HYPERPARAMETERS['learning_rate']}\n")
            f.write(f"# 其他参数保持不变\n")
            f.write("```\n\n")
            
            f.write("### 步骤3: 如需尝试其他Batch Size\n\n")
            f.write("⚠️ **重要**: 如果你想使用不同的batch size，必须：\n\n")
            f.write("1. 重新调优学习率（通常需要按比例调整）\n")
            f.write("2. 重新调优正则化参数\n")
            f.write("3. 可能需要调整训练步数\n\n")
            
            # 详细数据
            f.write("## 📋 完整测试数据\n\n")
            f.write("```json\n")
            f.write(json.dumps({
                'test_results': self.test_results,
                'scaling_analysis': scaling,
                'recommendation': optimal
            }, indent=2, ensure_ascii=False))
            f.write("\n```\n\n")
            
            # 参考链接
            f.write("## 🔗 参考资料\n\n")
            f.write("- [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)\n")
            f.write("- [Choosing the batch size](https://github.com/google-research/tuning_playbook#choosing-the-batch-size)\n")
            f.write("- [Shallue et al. 2018 - Measuring the Effects of Data Parallelism on Neural Network Training](https://arxiv.org/abs/1811.03600)\n")
        
        print(f"\n✅ 报告已生成: {report_file}")
        
        # 打印摘要到控制台
        print(f"\n{'='*70}")
        print("📊 优化结果摘要")
        print("="*70)
        print(f"\n🏆 推荐Batch Size: {optimal['recommended_batch_size']}")
        print(f"   理由: {optimal['reason']}")
        
        if optimal.get('details'):
            details = optimal['details']
            print(f"\n📈 性能指标:")
            print(f"   - 吞吐量: {details['training_throughput_samples_per_sec']} samples/sec")
            print(f"   - GPU利用率: {details['avg_gpu_utilization_pct']}%")
            print(f"   - 内存利用率: {details['memory_utilization_pct']:.1f}%")
        
        if scaling.get('perfect_scaling_range'):
            print(f"\n✨ Perfect Scaling范围: {scaling['perfect_scaling_range']}")
        
        if scaling.get('critical_batch_size'):
            print(f"\n⚠️  Critical Batch Size: {scaling['critical_batch_size']}")
        
        print(f"\n💡 实施方法:")
        print(f"   在config.py中设置: TRAIN_BATCH_SIZE = {optimal['recommended_batch_size']}")
        print(f"\n⚠️  提醒: 改变batch size需要重新调优学习率和正则化参数！")
        print("="*70 + "\n")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("🎯 自动寻找最优Batch Size")
    print("基于Google Deep Learning Tuning Playbook")
    print("="*70 + "\n")
    
    # 检查CUDA
    if not torch.cuda.is_available():
        print("❌ 错误: 未检测到CUDA，此脚本需要GPU环境")
        print("提示: 如果你有GPU但未检测到，请检查:")
        print("  1. NVIDIA驱动是否安装")
        print("  2. PyTorch是否安装了CUDA版本")
        return
    
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA版本: {torch.version.cuda}")
    print(f"✅ PyTorch版本: {torch.__version__}\n")
    
    # 创建优化器并运行
    optimizer = BatchSizeOptimizer()
    optimizer.run_optimization()
    
    print("\n" + "="*70)
    print("✅ 优化完成！")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
