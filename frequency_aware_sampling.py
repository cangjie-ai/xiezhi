"""
频率感知的采样策略

核心思想：
1. 保留频率信息（重复次数）
2. 高频查询 = 重要 → 多采样
3. 低频查询 = 长尾 → 少采但要覆盖
4. 平衡：80%根据频率加权，20%保证多样性

适用场景：
- Log数据，包含大量重复
- 追求实际生产环境的高F1
- 需要平衡主流需求和长尾覆盖
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter


class FrequencyAwareSampler:
    """
    频率感知采样器
    
    策略：
    1. 统计每条文本的出现频率
    2. 计算频率权重
    3. 分两阶段采样：
       - 阶段1（80%）：按频率加权采样（重要的多采）
       - 阶段2（20%）：均匀采样（保证长尾覆盖）
    """
    
    def __init__(
        self,
        frequency_ratio: float = 0.8,  # 频率加权采样的比例
        min_frequency: int = 1,         # 最小频率（过滤噪声）
        max_frequency_cap: int = None,  # 频率上限（避免单个样本权重过大）
        smoothing: str = "sqrt"          # 平滑方法: "sqrt", "log", "linear"
    ):
        """
        初始化频率感知采样器
        
        参数:
        - frequency_ratio: 按频率加权采样的比例（0-1），剩余的用于多样性采样
        - min_frequency: 最小频率阈值，低于此频率的可能是噪声
        - max_frequency_cap: 频率上限，避免某个超高频样本权重过大
        - smoothing: 频率平滑方法
            - "sqrt": 使用sqrt(freq)作为权重（推荐，平衡性好）
            - "log": 使用log(freq+1)作为权重（更激进的平滑）
            - "linear": 直接使用freq作为权重（保持原始分布）
        """
        self.frequency_ratio = frequency_ratio
        self.min_frequency = min_frequency
        self.max_frequency_cap = max_frequency_cap
        self.smoothing = smoothing
        
        print(f"频率感知采样器初始化:")
        print(f"  频率加权比例: {frequency_ratio:.0%}")
        print(f"  多样性保证比例: {1-frequency_ratio:.0%}")
        print(f"  频率平滑方法: {smoothing}")
    
    def compute_frequency_and_deduplicate(
        self,
        texts: List[str],
        original_indices: List[int] = None
    ) -> Tuple[List[str], List[int], List[int]]:
        """
        计算频率并去重
        
        参数:
        - texts: 文本列表（可能有重复）
        - original_indices: 原始索引列表
        
        返回:
        - unique_texts: 去重后的文本列表
        - unique_indices: 去重后的原始索引
        - frequencies: 每个唯一文本的出现频率
        """
        print(f"\n计算文本频率...")
        print(f"  原始数据: {len(texts)} 条")
        
        # 统计频率
        text_freq = Counter(texts)
        
        # 创建DataFrame
        if original_indices is None:
            original_indices = list(range(len(texts)))
        
        df = pd.DataFrame({
            'text': texts,
            'original_index': original_indices
        })
        
        # 去重，但保留第一次出现的索引
        unique_df = df.drop_duplicates(subset=['text'], keep='first').copy()
        
        # 添加频率信息
        unique_df['frequency'] = unique_df['text'].map(text_freq)
        
        # 过滤低频噪声
        if self.min_frequency > 1:
            before_filter = len(unique_df)
            unique_df = unique_df[unique_df['frequency'] >= self.min_frequency]
            print(f"  过滤低频(<{self.min_frequency})噪声: {before_filter} → {len(unique_df)} 条")
        
        # 应用频率上限
        if self.max_frequency_cap is not None:
            unique_df['frequency'] = unique_df['frequency'].clip(upper=self.max_frequency_cap)
        
        # 频率统计
        freq_values = unique_df['frequency'].values
        print(f"  唯一文本: {len(unique_df)} 条")
        print(f"  频率分布: min={freq_values.min()}, max={freq_values.max()}, mean={freq_values.mean():.1f}")
        print(f"  高频样本(>=10次): {np.sum(freq_values >= 10)} 条")
        print(f"  中频样本(2-9次): {np.sum((freq_values >= 2) & (freq_values < 10))} 条")
        print(f"  低频样本(1次): {np.sum(freq_values == 1)} 条")
        
        unique_texts = unique_df['text'].tolist()
        unique_indices = unique_df['original_index'].tolist()
        frequencies = unique_df['frequency'].tolist()
        
        return unique_texts, unique_indices, frequencies
    
    def compute_sampling_weights(self, frequencies: List[int]) -> np.ndarray:
        """
        计算采样权重
        
        参数:
        - frequencies: 频率列表
        
        返回:
        - weights: 归一化的采样权重
        """
        frequencies = np.array(frequencies, dtype=float)
        
        # 应用平滑
        if self.smoothing == "sqrt":
            weights = np.sqrt(frequencies)
        elif self.smoothing == "log":
            weights = np.log(frequencies + 1)
        elif self.smoothing == "linear":
            weights = frequencies
        else:
            raise ValueError(f"未知的平滑方法: {self.smoothing}")
        
        # 归一化
        weights = weights / weights.sum()
        
        return weights
    
    def frequency_aware_sample(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        original_indices: List[int],
        frequencies: List[int],
        n_samples: int,
        cluster_labels: np.ndarray = None
    ) -> Tuple[List[str], np.ndarray, List[int], List[int]]:
        """
        频率感知采样
        
        参数:
        - texts: 唯一文本列表
        - embeddings: 对应的embedding
        - original_indices: 原始索引
        - frequencies: 频率列表
        - n_samples: 目标采样数量
        - cluster_labels: 聚类标签（可选，用于多样性保证）
        
        返回:
        - sampled_texts: 采样的文本
        - sampled_embeddings: 采样的embedding
        - sampled_indices: 采样的原始索引
        - sampled_frequencies: 采样的频率
        """
        print(f"\n频率感知采样 (目标: {n_samples}条)...")
        
        n_total = len(texts)
        
        # 计算两个阶段的采样数量
        n_frequency_based = int(n_samples * self.frequency_ratio)
        n_diversity_based = n_samples - n_frequency_based
        
        print(f"  阶段1 (频率加权): {n_frequency_based} 条 ({self.frequency_ratio:.0%})")
        print(f"  阶段2 (多样性保证): {n_diversity_based} 条 ({1-self.frequency_ratio:.0%})")
        
        sampled_indices_set = set()
        
        # ===== 阶段1: 频率加权采样 =====
        print(f"\n  执行阶段1: 频率加权采样...")
        
        # 计算采样权重
        weights = self.compute_sampling_weights(frequencies)
        
        # 加权采样
        frequency_sampled_indices = np.random.choice(
            n_total,
            size=min(n_frequency_based, n_total),
            replace=False,
            p=weights
        )
        
        sampled_indices_set.update(frequency_sampled_indices)
        
        # 统计阶段1采样的频率分布
        stage1_freqs = [frequencies[i] for i in frequency_sampled_indices]
        print(f"    采样的频率分布: mean={np.mean(stage1_freqs):.1f}, median={np.median(stage1_freqs):.1f}")
        print(f"    高频样本占比: {np.sum(np.array(stage1_freqs) >= 10) / len(stage1_freqs):.1%}")
        
        # ===== 阶段2: 多样性保证采样 =====
        if n_diversity_based > 0:
            print(f"\n  执行阶段2: 多样性保证采样...")
            
            # 剩余未采样的索引
            remaining_indices = list(set(range(n_total)) - sampled_indices_set)
            
            if cluster_labels is not None:
                # 使用聚类信息进行分层采样
                diversity_sampled = self._stratified_diversity_sample(
                    remaining_indices,
                    cluster_labels,
                    n_diversity_based
                )
            else:
                # 简单均匀采样
                diversity_sampled = np.random.choice(
                    remaining_indices,
                    size=min(n_diversity_based, len(remaining_indices)),
                    replace=False
                )
            
            sampled_indices_set.update(diversity_sampled)
            
            # 统计阶段2采样的频率分布
            stage2_freqs = [frequencies[i] for i in diversity_sampled]
            print(f"    采样的频率分布: mean={np.mean(stage2_freqs):.1f}, median={np.median(stage2_freqs):.1f}")
            print(f"    低频样本占比: {np.sum(np.array(stage2_freqs) == 1) / len(stage2_freqs):.1%}")
        
        # ===== 整合结果 =====
        final_sampled_indices = np.array(list(sampled_indices_set))
        
        sampled_texts = [texts[i] for i in final_sampled_indices]
        sampled_embeddings = embeddings[final_sampled_indices]
        sampled_original_indices = [original_indices[i] for i in final_sampled_indices]
        sampled_frequencies = [frequencies[i] for i in final_sampled_indices]
        
        # 最终统计
        print(f"\n  ✓ 采样完成: {len(final_sampled_indices)} 条")
        print(f"    总频率覆盖: {sum(sampled_frequencies):,} / {sum(frequencies):,} = {sum(sampled_frequencies)/sum(frequencies):.1%}")
        print(f"    平均频率: {np.mean(sampled_frequencies):.1f}")
        print(f"    高频样本(>=10): {np.sum(np.array(sampled_frequencies) >= 10)} 条")
        print(f"    低频样本(=1): {np.sum(np.array(sampled_frequencies) == 1)} 条")
        
        return sampled_texts, sampled_embeddings, sampled_original_indices, sampled_frequencies
    
    def _stratified_diversity_sample(
        self,
        remaining_indices: List[int],
        cluster_labels: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        分层多样性采样（从每个簇中采样）
        """
        sampled = []
        
        # 获取剩余索引对应的簇标签
        remaining_labels = cluster_labels[remaining_indices]
        unique_labels = np.unique(remaining_labels)
        
        # 每个簇平均采样
        n_per_cluster = max(1, n_samples // len(unique_labels))
        
        for label in unique_labels:
            cluster_mask = (remaining_labels == label)
            cluster_indices = np.array(remaining_indices)[cluster_mask]
            
            if len(cluster_indices) > 0:
                n_sample = min(n_per_cluster, len(cluster_indices))
                selected = np.random.choice(cluster_indices, size=n_sample, replace=False)
                sampled.extend(selected)
        
        # 如果不够，随机补充
        if len(sampled) < n_samples:
            remaining = list(set(remaining_indices) - set(sampled))
            if len(remaining) > 0:
                additional = np.random.choice(
                    remaining,
                    size=min(n_samples - len(sampled), len(remaining)),
                    replace=False
                )
                sampled.extend(additional)
        
        # 截断到目标数量
        sampled = sampled[:n_samples]
        
        return np.array(sampled)


def analyze_frequency_distribution(texts: List[str]) -> Dict:
    """
    分析文本频率分布
    
    用于决策是否需要频率感知采样
    """
    print("分析频率分布...")
    
    freq_counter = Counter(texts)
    frequencies = list(freq_counter.values())
    
    # 统计
    total_texts = len(texts)
    unique_texts = len(freq_counter)
    dedup_rate = unique_texts / total_texts
    
    freq_array = np.array(frequencies)
    
    stats = {
        'total_texts': total_texts,
        'unique_texts': unique_texts,
        'deduplication_rate': dedup_rate,
        'frequency_stats': {
            'min': int(freq_array.min()),
            'max': int(freq_array.max()),
            'mean': float(freq_array.mean()),
            'median': float(np.median(freq_array)),
            'std': float(freq_array.std()),
        },
        'distribution': {
            'high_freq (>=10)': int(np.sum(freq_array >= 10)),
            'medium_freq (2-9)': int(np.sum((freq_array >= 2) & (freq_array < 10))),
            'low_freq (=1)': int(np.sum(freq_array == 1)),
        }
    }
    
    print(f"\n频率分布分析:")
    print(f"  总文本数: {total_texts:,}")
    print(f"  唯一文本: {unique_texts:,}")
    print(f"  去重率: {dedup_rate:.1%}")
    print(f"\n频率统计:")
    print(f"  最小: {stats['frequency_stats']['min']}")
    print(f"  最大: {stats['frequency_stats']['max']}")
    print(f"  平均: {stats['frequency_stats']['mean']:.1f}")
    print(f"  中位数: {stats['frequency_stats']['median']:.1f}")
    print(f"\n分布:")
    print(f"  高频(>=10次): {stats['distribution']['high_freq (>=10)']} ({stats['distribution']['high_freq (>=10)']/unique_texts:.1%})")
    print(f"  中频(2-9次): {stats['distribution']['medium_freq (2-9)']} ({stats['distribution']['medium_freq (2-9)']/unique_texts:.1%})")
    print(f"  低频(1次): {stats['distribution']['low_freq (=1)']} ({stats['distribution']['low_freq (=1)']/unique_texts:.1%})")
    
    # 建议
    print(f"\n💡 建议:")
    if dedup_rate < 0.5:
        print(f"  ✅ 去重率很低({dedup_rate:.1%})，重复度高，强烈建议使用频率感知采样")
    elif dedup_rate < 0.8:
        print(f"  ✅ 去重率中等({dedup_rate:.1%})，有一定重复，建议使用频率感知采样")
    else:
        print(f"  ⚠️ 去重率很高({dedup_rate:.1%})，重复度低，频率感知采样提升可能有限")
    
    if stats['frequency_stats']['max'] > 100:
        print(f"  ⚠️ 存在超高频样本(最高{stats['frequency_stats']['max']}次)，建议设置max_frequency_cap")
    
    return stats


# 使用示例
if __name__ == "__main__":
    print("""
使用示例：

# 1. 分析数据频率分布
import pandas as pd
df = pd.read_csv('data/cleaned_460k.csv')
texts = df['text'].tolist()

stats = analyze_frequency_distribution(texts)

# 2. 创建频率感知采样器
sampler = FrequencyAwareSampler(
    frequency_ratio=0.8,     # 80%按频率，20%保证多样性
    min_frequency=2,         # 过滤只出现1次的（可能是噪声）
    max_frequency_cap=1000,  # 频率上限，避免单个样本权重过大
    smoothing="sqrt"         # 使用sqrt平滑（推荐）
)

# 3. 计算频率并去重
unique_texts, unique_indices, frequencies = sampler.compute_frequency_and_deduplicate(texts)

# 4. 生成embeddings（假设已有）
# embeddings = embed_texts(unique_texts)

# 5. 频率感知采样
sampled_texts, sampled_embeddings, sampled_indices, sampled_frequencies = \
    sampler.frequency_aware_sample(
        texts=unique_texts,
        embeddings=embeddings,
        original_indices=unique_indices,
        frequencies=frequencies,
        n_samples=15000,
        cluster_labels=cluster_labels  # 可选
    )

# 6. 保存时包含频率信息
output_df = pd.DataFrame({
    'text': sampled_texts,
    'frequency': sampled_frequencies,  # 保留频率信息
    'original_index': sampled_indices,
    'importance': 'high' if freq >= 10 else 'medium' if freq >= 2 else 'low',
    'label': ''
})
""")

