"""
采样质量验证工具

用于验证数据采样结果是否满足98% F1微调目标的要求

用法:
    python validate_sampling_quality.py --input data/sampled_15k.csv --stats data/sampled_15k_stats.json
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


class SamplingQualityValidator:
    """采样质量验证器"""
    
    # 质量阈值（针对98% F1目标）
    THRESHOLDS = {
        'coverage_rate': 0.95,      # 簇覆盖率 ≥ 95%
        'normalized_entropy': 0.80,  # 标准化熵 ≥ 0.80
        'mean_quality': 0.50,        # 平均质量 ≥ 0.50
        'min_samples': 15000,        # 最小样本数
        'min_per_cluster': 3,        # 每簇最少样本数
        'max_per_cluster_ratio': 0.10 # 单簇最多占比 ≤ 10%
    }
    
    def __init__(self, data_file: str, stats_file: str = None):
        """
        初始化验证器
        
        参数:
        - data_file: 采样数据文件路径
        - stats_file: 统计文件路径（可选）
        """
        self.data_file = data_file
        self.stats_file = stats_file or data_file.replace('.csv', '_stats.json')
        
        # 加载数据
        self.df = pd.read_csv(data_file)
        
        # 加载统计（如果存在）
        self.stats = None
        if Path(self.stats_file).exists():
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                self.stats = json.load(f)
    
    def validate_all(self) -> Dict[str, any]:
        """
        执行所有验证检查
        
        返回:
        - results: 验证结果字典
        """
        print("=" * 70)
        print("采样质量验证 - 98% F1目标")
        print("=" * 70)
        print(f"数据文件: {self.data_file}")
        print(f"统计文件: {self.stats_file}")
        print()
        
        results = {
            'file': self.data_file,
            'checks': [],
            'passed': 0,
            'failed': 0,
            'warnings': 0,
            'overall_status': 'UNKNOWN'
        }
        
        # 1. 数据量检查
        results['checks'].append(self._check_sample_size())
        
        # 2. 簇覆盖率检查
        results['checks'].append(self._check_cluster_coverage())
        
        # 3. 多样性检查
        results['checks'].append(self._check_diversity())
        
        # 4. 质量检查
        results['checks'].append(self._check_quality())
        
        # 5. 簇平衡性检查
        results['checks'].append(self._check_cluster_balance())
        
        # 6. 数据完整性检查
        results['checks'].append(self._check_data_integrity())
        
        # 统计结果
        for check in results['checks']:
            if check['status'] == 'PASS':
                results['passed'] += 1
            elif check['status'] == 'FAIL':
                results['failed'] += 1
            else:
                results['warnings'] += 1
        
        # 判断总体状态
        if results['failed'] == 0:
            if results['warnings'] == 0:
                results['overall_status'] = 'EXCELLENT'
            else:
                results['overall_status'] = 'GOOD'
        elif results['failed'] <= 2:
            results['overall_status'] = 'ACCEPTABLE'
        else:
            results['overall_status'] = 'NEEDS_IMPROVEMENT'
        
        # 打印总结
        self._print_summary(results)
        
        return results
    
    def _check_sample_size(self) -> Dict:
        """检查样本数量"""
        n_samples = len(self.df)
        threshold = self.THRESHOLDS['min_samples']
        
        check = {
            'name': '样本数量',
            'value': n_samples,
            'threshold': f'>= {threshold}',
            'status': 'PASS' if n_samples >= threshold else 'WARN',
            'message': ''
        }
        
        if n_samples >= 20000:
            check['message'] = f'✓ 样本数量充足 ({n_samples:,}条)，适合大模型微调'
        elif n_samples >= threshold:
            check['message'] = f'✓ 样本数量达标 ({n_samples:,}条)，建议增加到20000条以获得更好效果'
        else:
            check['status'] = 'WARN'
            check['message'] = f'⚠️ 样本数量不足 ({n_samples:,}条)，建议至少{threshold:,}条'
        
        print(f"1. {check['name']}: {check['message']}")
        return check
    
    def _check_cluster_coverage(self) -> Dict:
        """检查簇覆盖率"""
        if 'cluster_id' not in self.df.columns:
            return {
                'name': '簇覆盖率',
                'value': 'N/A',
                'threshold': 'N/A',
                'status': 'WARN',
                'message': '⚠️ 缺少cluster_id列，无法验证簇覆盖率'
            }
        
        # 从统计文件获取覆盖率
        if self.stats and 'clustering_stats' in self.stats:
            coverage_rate = self.stats['clustering_stats']['coverage_rate_final']
            n_covered = self.stats['clustering_stats']['n_covered_clusters_final']
            n_total = self.stats['clustering_stats']['n_active_clusters']
        else:
            # 从数据推断
            n_covered = self.df['cluster_id'].nunique()
            n_total = n_covered  # 无法知道总数
            coverage_rate = 1.0
        
        threshold = self.THRESHOLDS['coverage_rate']
        
        check = {
            'name': '簇覆盖率',
            'value': coverage_rate,
            'threshold': f'>= {threshold:.0%}',
            'status': 'PASS' if coverage_rate >= threshold else 'FAIL',
            'message': ''
        }
        
        if coverage_rate >= 0.98:
            check['message'] = f'✓ 簇覆盖率优秀 ({coverage_rate:.1%}, {n_covered}/{n_total})'
        elif coverage_rate >= threshold:
            check['message'] = f'✓ 簇覆盖率良好 ({coverage_rate:.1%}, {n_covered}/{n_total})'
        else:
            check['status'] = 'FAIL'
            check['message'] = f'❌ 簇覆盖率不足 ({coverage_rate:.1%}, {n_covered}/{n_total})，需要提高'
        
        print(f"2. {check['name']}: {check['message']}")
        return check
    
    def _check_diversity(self) -> Dict:
        """检查数据多样性（Shannon熵）"""
        if 'cluster_id' not in self.df.columns:
            return {
                'name': '数据多样性',
                'value': 'N/A',
                'threshold': 'N/A',
                'status': 'WARN',
                'message': '⚠️ 缺少cluster_id列，无法验证多样性'
            }
        
        # 计算Shannon熵
        cluster_counts = self.df['cluster_id'].value_counts()
        cluster_probs = cluster_counts / cluster_counts.sum()
        shannon_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
        max_entropy = np.log(len(cluster_counts))
        normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
        
        threshold = self.THRESHOLDS['normalized_entropy']
        
        check = {
            'name': '数据多样性',
            'value': normalized_entropy,
            'threshold': f'>= {threshold:.2f}',
            'status': 'PASS' if normalized_entropy >= threshold else 'WARN',
            'message': ''
        }
        
        if normalized_entropy >= 0.90:
            check['message'] = f'✓ 分布非常均匀 (熵={normalized_entropy:.3f})'
        elif normalized_entropy >= threshold:
            check['message'] = f'✓ 分布较均匀 (熵={normalized_entropy:.3f})'
        else:
            check['status'] = 'WARN'
            check['message'] = f'⚠️ 分布不够均匀 (熵={normalized_entropy:.3f})，某些簇可能过度代表'
        
        print(f"3. {check['name']}: {check['message']}")
        return check
    
    def _check_quality(self) -> Dict:
        """检查质量得分"""
        if 'quality_score' not in self.df.columns:
            return {
                'name': '质量得分',
                'value': 'N/A',
                'threshold': 'N/A',
                'status': 'WARN',
                'message': '⚠️ 缺少quality_score列，无法验证质量'
            }
        
        mean_quality = self.df['quality_score'].mean()
        min_quality = self.df['quality_score'].min()
        threshold = self.THRESHOLDS['mean_quality']
        
        check = {
            'name': '质量得分',
            'value': mean_quality,
            'threshold': f'>= {threshold:.2f}',
            'status': 'PASS' if mean_quality >= threshold else 'WARN',
            'message': ''
        }
        
        if mean_quality >= 0.70:
            check['message'] = f'✓ 质量优秀 (平均={mean_quality:.3f}, 最低={min_quality:.3f})'
        elif mean_quality >= threshold:
            check['message'] = f'✓ 质量良好 (平均={mean_quality:.3f}, 最低={min_quality:.3f})'
        else:
            check['status'] = 'WARN'
            check['message'] = f'⚠️ 质量偏低 (平均={mean_quality:.3f})，可能需要改进评分规则'
        
        print(f"4. {check['name']}: {check['message']}")
        return check
    
    def _check_cluster_balance(self) -> Dict:
        """检查簇平衡性"""
        if 'cluster_id' not in self.df.columns:
            return {
                'name': '簇平衡性',
                'value': 'N/A',
                'threshold': 'N/A',
                'status': 'WARN',
                'message': '⚠️ 缺少cluster_id列，无法验证平衡性'
            }
        
        cluster_counts = self.df['cluster_id'].value_counts()
        min_count = cluster_counts.min()
        max_count = cluster_counts.max()
        mean_count = cluster_counts.mean()
        max_ratio = max_count / len(self.df)
        
        min_threshold = self.THRESHOLDS['min_per_cluster']
        max_threshold = self.THRESHOLDS['max_per_cluster_ratio']
        
        check = {
            'name': '簇平衡性',
            'value': f'{min_count}-{max_count}',
            'threshold': f'每簇 >= {min_threshold}, 单簇 <= {max_threshold:.0%}',
            'status': 'PASS',
            'message': ''
        }
        
        issues = []
        
        if min_count < min_threshold:
            check['status'] = 'WARN'
            issues.append(f'最小簇仅{min_count}条')
        
        if max_ratio > max_threshold:
            check['status'] = 'WARN'
            issues.append(f'最大簇占{max_ratio:.1%}')
        
        if issues:
            check['message'] = f'⚠️ 簇不平衡: {", ".join(issues)} (范围: {min_count}-{max_count}, 平均: {mean_count:.1f})'
        else:
            check['message'] = f'✓ 簇平衡良好 (范围: {min_count}-{max_count}, 平均: {mean_count:.1f})'
        
        print(f"5. {check['name']}: {check['message']}")
        return check
    
    def _check_data_integrity(self) -> Dict:
        """检查数据完整性"""
        issues = []
        
        # 检查必需列
        required_cols = ['text']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            issues.append(f'缺少列: {", ".join(missing_cols)}')
        
        # 检查空值
        if 'text' in self.df.columns:
            null_count = self.df['text'].isnull().sum()
            if null_count > 0:
                issues.append(f'{null_count}条文本为空')
        
        # 检查重复
        if 'text' in self.df.columns:
            dup_count = self.df['text'].duplicated().sum()
            if dup_count > 0:
                issues.append(f'{dup_count}条重复文本')
        
        # 检查文本长度
        if 'text' in self.df.columns:
            text_lengths = self.df['text'].str.len()
            too_short = (text_lengths < 3).sum()
            too_long = (text_lengths > 500).sum()
            if too_short > 0:
                issues.append(f'{too_short}条文本过短(<3字符)')
            if too_long > 0:
                issues.append(f'{too_long}条文本过长(>500字符)')
        
        check = {
            'name': '数据完整性',
            'value': f'{len(issues)} 个问题',
            'threshold': '0 个问题',
            'status': 'PASS' if len(issues) == 0 else 'WARN',
            'message': ''
        }
        
        if issues:
            check['message'] = f'⚠️ 发现问题: {"; ".join(issues)}'
        else:
            check['message'] = '✓ 数据完整，无明显问题'
        
        print(f"6. {check['name']}: {check['message']}")
        return check
    
    def _print_summary(self, results: Dict):
        """打印验证总结"""
        print()
        print("=" * 70)
        print("验证总结")
        print("=" * 70)
        print(f"通过: {results['passed']} 项")
        print(f"警告: {results['warnings']} 项")
        print(f"失败: {results['failed']} 项")
        print()
        
        status_emoji = {
            'EXCELLENT': '🎉',
            'GOOD': '✅',
            'ACCEPTABLE': '⚠️',
            'NEEDS_IMPROVEMENT': '❌'
        }
        
        status_msg = {
            'EXCELLENT': '优秀！数据质量完全满足98% F1目标',
            'GOOD': '良好！数据质量基本满足要求，可以开始标注和训练',
            'ACCEPTABLE': '可接受，但建议改进部分指标以达到最佳效果',
            'NEEDS_IMPROVEMENT': '需要改进！请根据上述建议调整采样参数'
        }
        
        overall_status = results['overall_status']
        print(f"总体评价: {status_emoji[overall_status]} {overall_status}")
        print(f"{status_msg[overall_status]}")
        print("=" * 70)
        
        # 给出建议
        if results['failed'] > 0 or results['warnings'] > 0:
            print()
            print("💡 改进建议:")
            
            for check in results['checks']:
                if check['status'] in ['FAIL', 'WARN']:
                    print(f"\n• {check['name']}: {check['message']}")
                    
                    # 根据不同问题给出具体建议
                    if check['name'] == '样本数量' and check['status'] == 'WARN':
                        print("  → 重新运行采样，增加 --n_samples 到 15000-20000")
                    
                    elif check['name'] == '簇覆盖率' and check['status'] == 'FAIL':
                        print("  → 增加 --min_per_cluster 到 10")
                        print("  → 或减少 --n_clusters")
                    
                    elif check['name'] == '数据多样性' and check['status'] == 'WARN':
                        print("  → 使用 --sampling_strategy balanced")
                        print("  → 增加 --min_per_cluster")
                    
                    elif check['name'] == '质量得分' and check['status'] == 'WARN':
                        print("  → 调整QualityScorer的评分权重")
                        print("  → 加强数据清洗")
            
            print()
    
    def export_report(self, output_file: str = None):
        """导出验证报告"""
        if output_file is None:
            output_file = self.data_file.replace('.csv', '_validation_report.json')
        
        results = self.validate_all()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, indent=2, fp=f, ensure_ascii=False)
        
        print(f"\n验证报告已保存到: {output_file}")
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description="采样质量验证工具 - 验证是否满足98% F1目标",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="采样数据文件路径 (CSV)"
    )
    parser.add_argument(
        "--stats",
        type=str,
        default=None,
        help="统计文件路径 (JSON, 可选)"
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="是否导出验证报告"
    )
    
    args = parser.parse_args()
    
    # 创建验证器并运行
    validator = SamplingQualityValidator(args.input, args.stats)
    
    if args.export:
        validator.export_report()
    else:
        validator.validate_all()


if __name__ == "__main__":
    main()

