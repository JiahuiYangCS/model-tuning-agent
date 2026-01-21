"""
GPU Batch Size优化测试脚本

遵循Google Deep Learning Tuning Playbook原则：
1. Batch size不影响最终模型精度，只影响训练速度和资源利用
2. 目标是找到能充分利用GPU的最优batch size
3. 固定所有其他超参数（学习率、优化器等）
4. 测试不同batch size下的GPU利用率、吞吐量、训练时间

使用方法：
    python scripts/optimize_gpu_batch_size.py
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# 添加项目根目录到sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from gpu_monitor.pynvml_monitor import PyNVMLMonitor


class GPUBatchSizeOptimizer:
    """GPU Batch Size优化器"""
    
    def __init__(self):
        self.gpu_monitor = PyNVMLMonitor()
        self.test_results = []
        
        # 固定训练参数（不会改变）
        self.FIXED_PARAMS = {
            'learning_rate': 2e-5,  # 固定学习率
            'epochs': 1,  # 测试用1个epoch即可
            'warmup_steps': 100,
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'optimizer': 'AdamW',
            'weight_decay': 0.01,
            'max_grad_norm': 1.0,
        }
        
        # 要测试的batch size列表（从小到大）
        self.BATCH_SIZES_TO_TEST = [8, 16, 24, 32, 48, 64, 96, 128]
        
        # 测试数据集大小（固定）
        self.NUM_SAMPLES = 1000
        
        self.report_path = project_root / 'docs' / 'reports'
        self.report_path.mkdir(parents=True, exist_ok=True)
    
    def create_dummy_dataset(self, num_samples=1000):
        """创建固定的测试数据集"""
        print(f"📊 创建测试数据集：{num_samples}个样本")
        
        examples = []
        for i in range(num_samples):
            text1 = f"This is a sample training text number {i} for testing purposes."
            text2 = f"This is another related text number {i} for similarity training."
            label = 0.8  # 固定相似度标签
            examples.append(InputExample(texts=[text1, text2], label=label))
        
        return examples
    
    def test_batch_size(self, batch_size):
        """测试单个batch size的性能"""
        print(f"\n{'='*70}")
        print(f"🧪 测试 Batch Size = {batch_size}")
        print(f"{'='*70}")
        
        try:
            # 清空GPU缓存
            torch.cuda.empty_cache()
            time.sleep(2)
            
            # 记录初始GPU状态
            initial_stats = self.gpu_monitor.get_instant_metrics()
            initial_memory = initial_stats.get('memory_used_mb', 0)
            
            # 加载模型
            print("🔧 加载模型...")
            model = SentenceTransformer(self.FIXED_PARAMS['model_name'])
            model.to('cuda')
            
            # 创建数据集
            train_examples = self.create_dummy_dataset(self.NUM_SAMPLES)
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size
            )
            
            # 定义loss
            train_loss = losses.CosineSimilarityLoss(model)
            
            # 记录训练前GPU状态
            before_training_stats = self.gpu_monitor.get_instant_metrics()
            
            # 开始训练并计时
            print(f"🚀 开始训练（固定学习率={self.FIXED_PARAMS['learning_rate']}）...")
            start_time = time.time()
            
            # 记录GPU监控数据
            gpu_samples = []
            
            # 训练（只训练1个epoch用于测试）
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=self.FIXED_PARAMS['epochs'],
                warmup_steps=self.FIXED_PARAMS['warmup_steps'],
                optimizer_params={'lr': self.FIXED_PARAMS['learning_rate']},
                show_progress_bar=True,
            )
            
            # 在训练过程中采样GPU状态
            for _ in range(5):
                gpu_samples.append(self.gpu_monitor.get_instant_metrics())
                time.sleep(0.5)
            
            end_time = time.time()
            training_time = end_time - start_time
            
            # 训练后GPU状态
            after_training_stats = self.gpu_monitor.get_instant_metrics()
            
            # 计算吞吐量
            samples_per_second = self.NUM_SAMPLES / training_time
            batches_per_second = len(train_dataloader) / training_time
            
            # 计算平均GPU利用率
            avg_gpu_util = sum(s.get('gpu_utilization', 0) for s in gpu_samples) / len(gpu_samples)
            avg_memory_used = sum(s.get('memory_used_mb', 0) for s in gpu_samples) / len(gpu_samples)
            peak_memory_used = max(s.get('memory_used_mb', 0) for s in gpu_samples)
            
            # 计算内存利用率
            total_memory = self.gpu_monitor.get_instant_metrics().get('memory_total_mb', 1)
            memory_utilization_pct = (peak_memory_used / total_memory) * 100
            
            result = {
                'batch_size': batch_size,
                'success': True,
                'training_time_seconds': round(training_time, 2),
                'samples_per_second': round(samples_per_second, 2),
                'batches_per_second': round(batches_per_second, 2),
                'avg_gpu_utilization_pct': round(avg_gpu_util, 2),
                'peak_memory_used_mb': round(peak_memory_used, 2),
                'avg_memory_used_mb': round(avg_memory_used, 2),
                'memory_utilization_pct': round(memory_utilization_pct, 2),
                'total_memory_mb': total_memory,
                'num_batches': len(train_dataloader),
                'time_per_batch_ms': round((training_time / len(train_dataloader)) * 1000, 2),
            }
            
            print(f"\n✅ 测试完成")
            print(f"   训练时间: {result['training_time_seconds']}秒")
            print(f"   吞吐量: {result['samples_per_second']} samples/sec")
            print(f"   平均GPU利用率: {result['avg_gpu_utilization_pct']}%")
            print(f"   峰值内存使用: {result['peak_memory_used_mb']} MB ({result['memory_utilization_pct']:.1f}%)")
            
            # 清理
            del model
            torch.cuda.empty_cache()
            time.sleep(2)
            
            return result
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"❌ Batch Size {batch_size} 超出GPU内存限制")
                return {
                    'batch_size': batch_size,
                    'success': False,
                    'error': 'OOM (Out of Memory)',
                    'message': f'Batch size {batch_size}超出GPU内存'
                }
            else:
                print(f"❌ 测试失败: {e}")
                return {
                    'batch_size': batch_size,
                    'success': False,
                    'error': str(e)
                }
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return {
                'batch_size': batch_size,
                'success': False,
                'error': str(e)
            }
    
    def run_optimization(self):
        """运行完整的batch size优化测试"""
        print("\n" + "="*70)
        print("🎯 GPU Batch Size优化测试")
        print("="*70)
        print(f"\n📋 固定训练参数:")
        for key, value in self.FIXED_PARAMS.items():
            print(f"   {key}: {value}")
        print(f"\n🧪 测试Batch Sizes: {self.BATCH_SIZES_TO_TEST}")
        print(f"📊 测试数据集大小: {self.NUM_SAMPLES} samples")
        print(f"\n⚡ 开始测试...\n")
        
        # 记录GPU信息
        gpu_info = self.gpu_monitor.get_instant_metrics()
        
        # 测试每个batch size
        for batch_size in self.BATCH_SIZES_TO_TEST:
            result = self.test_batch_size(batch_size)
            self.test_results.append(result)
            
            # 如果遇到OOM，停止测试更大的batch size
            if not result['success'] and result.get('error') == 'OOM (Out of Memory)':
                print(f"\n⚠️  达到GPU内存上限，停止测试更大的batch size")
                break
            
            # 短暂休息
            time.sleep(3)
        
        # 生成报告
        self.generate_report(gpu_info)
    
    def generate_report(self, gpu_info):
        """生成优化报告"""
        print(f"\n\n{'='*70}")
        print("📊 生成优化报告")
        print("="*70)
        
        # 过滤成功的测试
        successful_tests = [r for r in self.test_results if r['success']]
        
        if not successful_tests:
            print("❌ 没有成功的测试结果")
            return
        
        # 找到最优batch size（基于多个指标）
        # 1. 吞吐量最高
        best_throughput = max(successful_tests, key=lambda x: x['samples_per_second'])
        
        # 2. GPU利用率最高
        best_gpu_util = max(successful_tests, key=lambda x: x['avg_gpu_utilization_pct'])
        
        # 3. 内存利用率最高（但不超过90%，留有余地）
        safe_memory_tests = [r for r in successful_tests if r['memory_utilization_pct'] <= 90]
        if safe_memory_tests:
            best_memory_util = max(safe_memory_tests, key=lambda x: x['memory_utilization_pct'])
        else:
            best_memory_util = successful_tests[-1]  # 最大成功的batch size
        
        # 4. 综合评分（加权平均）
        for result in successful_tests:
            # 归一化各项指标（0-100分）
            max_throughput = max(r['samples_per_second'] for r in successful_tests)
            max_gpu_util = max(r['avg_gpu_utilization_pct'] for r in successful_tests)
            
            throughput_score = (result['samples_per_second'] / max_throughput) * 100
            gpu_util_score = result['avg_gpu_utilization_pct']
            memory_score = min(result['memory_utilization_pct'], 100)
            
            # 综合评分：吞吐量40%，GPU利用率30%，内存利用率30%
            result['综合评分'] = (
                throughput_score * 0.4 +
                gpu_util_score * 0.3 +
                memory_score * 0.3
            )
        
        best_overall = max(successful_tests, key=lambda x: x['综合评分'])
        
        # 生成Markdown报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.report_path / f"batch_size_optimization_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# GPU Batch Size优化报告\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # GPU信息
            f.write("## 🖥️ GPU信息\n\n")
            f.write(f"- **GPU型号**: {gpu_info.get('name', 'Unknown')}\n")
            f.write(f"- **显存总量**: {gpu_info.get('memory_total_mb', 0):.0f} MB\n")
            f.write(f"- **CUDA版本**: {torch.version.cuda}\n\n")
            
            # 固定参数
            f.write("## 🔧 固定训练参数\n\n")
            f.write("根据Google Deep Learning Tuning Playbook原则，除batch size外的所有参数固定：\n\n")
            for key, value in self.FIXED_PARAMS.items():
                f.write(f"- **{key}**: `{value}`\n")
            f.write(f"- **测试数据集大小**: {self.NUM_SAMPLES} samples\n\n")
            
            # 测试结果表格
            f.write("## 📊 测试结果对比\n\n")
            f.write("| Batch Size | 状态 | 训练时间(s) | 吞吐量(samples/s) | GPU利用率(%) | 峰值内存(MB) | 内存利用率(%) | 综合评分 |\n")
            f.write("|------------|------|-------------|-------------------|--------------|--------------|---------------|----------|\n")
            
            for result in self.test_results:
                if result['success']:
                    f.write(
                        f"| {result['batch_size']} | ✅ | "
                        f"{result['training_time_seconds']} | "
                        f"{result['samples_per_second']} | "
                        f"{result['avg_gpu_utilization_pct']} | "
                        f"{result['peak_memory_used_mb']:.0f} | "
                        f"{result['memory_utilization_pct']:.1f} | "
                        f"{result['综合评分']:.1f} |\n"
                    )
                else:
                    f.write(f"| {result['batch_size']} | ❌ | - | - | - | - | - | - |\n")
            
            # 推荐结果
            f.write("\n## 🎯 优化推荐\n\n")
            
            f.write("### 🏆 最佳综合性能\n\n")
            f.write(f"**推荐Batch Size: {best_overall['batch_size']}**\n\n")
            f.write(f"- 综合评分: {best_overall['综合评分']:.1f}/100\n")
            f.write(f"- 吞吐量: {best_overall['samples_per_second']} samples/sec\n")
            f.write(f"- GPU利用率: {best_overall['avg_gpu_utilization_pct']}%\n")
            f.write(f"- 内存利用率: {best_overall['memory_utilization_pct']:.1f}%\n")
            f.write(f"- 训练时间: {best_overall['training_time_seconds']}秒\n\n")
            
            f.write("### 📈 其他优化指标\n\n")
            f.write(f"- **最高吞吐量**: Batch Size = {best_throughput['batch_size']} "
                   f"({best_throughput['samples_per_second']} samples/sec)\n")
            f.write(f"- **最高GPU利用率**: Batch Size = {best_gpu_util['batch_size']} "
                   f"({best_gpu_util['avg_gpu_utilization_pct']}%)\n")
            f.write(f"- **最优内存利用**: Batch Size = {best_memory_util['batch_size']} "
                   f"({best_memory_util['memory_utilization_pct']:.1f}%)\n\n")
            
            # Google Playbook原则提醒
            f.write("## ⚠️ 重要提醒（Google Deep Learning Tuning Playbook）\n\n")
            f.write("1. **Batch size不影响最终模型精度**：只要其他超参数调优得当，任何batch size都能达到相同的最终性能\n")
            f.write("2. **改变batch size需要重新调整学习率**：如果你采用不同的batch size，必须重新调优学习率和其他超参数\n")
            f.write("3. **Batch size影响训练速度**：更大的batch size通常能更好地利用GPU，提高训练速度\n")
            f.write("4. **留有内存余地**：建议使用内存利用率在80-90%的batch size，避免OOM风险\n\n")
            
            # 实施建议
            f.write("## 🚀 实施建议\n\n")
            f.write(f"1. 在`config.py`中设置：\n")
            f.write(f"   ```python\n")
            f.write(f"   TRAIN_BATCH_SIZE = {best_overall['batch_size']}\n")
            f.write(f"   EVAL_BATCH_SIZE = {best_overall['batch_size']}\n")
            f.write(f"   ```\n\n")
            f.write(f"2. 保持当前学习率不变（因为测试时使用的就是这个学习率）\n\n")
            f.write(f"3. 如果要尝试其他batch size，记得重新调优学习率\n\n")
            
            # 详细数据
            f.write("## 📋 详细测试数据\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.test_results, indent=2, ensure_ascii=False))
            f.write("\n```\n")
        
        print(f"\n✅ 报告已生成: {report_file}")
        
        # 打印摘要到控制台
        print(f"\n{'='*70}")
        print("📊 优化结果摘要")
        print("="*70)
        print(f"\n🏆 推荐Batch Size: {best_overall['batch_size']}")
        print(f"   - 综合评分: {best_overall['综合评分']:.1f}/100")
        print(f"   - 吞吐量: {best_overall['samples_per_second']} samples/sec")
        print(f"   - GPU利用率: {best_overall['avg_gpu_utilization_pct']}%")
        print(f"   - 内存利用率: {best_overall['memory_utilization_pct']:.1f}%")
        print(f"\n💡 实施方法：在config.py中设置 TRAIN_BATCH_SIZE = {best_overall['batch_size']}")
        print(f"\n⚠️  提醒：根据Google Playbook原则，batch size只影响训练速度，不影响最终精度")
        print("="*70 + "\n")


def main():
    """主函数"""
    # 检查CUDA是否可用
    if not torch.cuda.is_available():
        print("❌ 错误：未检测到CUDA，此脚本需要GPU环境")
        return
    
    print(f"✅ 检测到GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ CUDA版本: {torch.version.cuda}")
    
    # 创建优化器并运行
    optimizer = GPUBatchSizeOptimizer()
    optimizer.run_optimization()


if __name__ == "__main__":
    main()
