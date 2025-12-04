"""
二次采样脚本 - 从已清洗数据中精选高质量样本

适用场景：
- 已经通过其他方法获得了清洗后的数据
- 想进一步采样高质量样本用于标注
- 需要保证多样性和覆盖度

用法：
    python secondary_sampling.py --input cleaned_40k.csv --output final_10k.csv --n_samples 10000 --n_clusters 300
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from typing import List, Tuple

# 导入主pipeline的组件
from data_sampling_pipeline import (
    TextEmbedder,
    DiversitySampler,
    QualityScorer,
    DataCleaner
)


class SecondarySampler:
    """
    二次采样器 - 专门用于从已清洗数据中精选样本
    
    流程：
    1. 加载数据（假设已清洗）
    2. 轻量清洗（可选，去除极端情况）
    3. 生成embeddings
    4. 聚类
    5. 分层采样
    6. 质量评分
    7. 输出
    """
    
    def __init__(
        self,
        input_file: str,
        output_file: str,
        n_samples: int = 10000,
        n_clusters: int = 300,
        embedding_model: str = "moka-ai/m3e-base",
        enable_light_cleaning: bool = True
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        self.embedding_model = embedding_model
        self.enable_light_cleaning = enable_light_cleaning
        
        print(f"二次采样配置:")
        print(f"  输入: {input_file}")
        print(f"  输出: {output_file}")
        print(f"  目标样本数: {n_samples}")
        print(f"  聚类数: {n_clusters}")
    
    def run(self):
        """执行二次采样"""
        
        print("\n" + "="*70)
        print("二次采样Pipeline启动")
        print("="*70)
        
        # 步骤1: 加载数据
        print("\n步骤1: 加载数据")
        print("-"*70)
        df = pd.read_csv(self.input_file)
        
        # 查找文本列
        text_column = None
        for col in ['text', 'query', 'question', 'content', 'message']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            raise ValueError(f"未找到文本列。可用列: {df.columns.tolist()}")
        
        texts = df[text_column].tolist()
        print(f"✓ 加载了 {len(texts)} 条文本")
        
        # 步骤2: 轻量清洗（可选）
        if self.enable_light_cleaning:
            print("\n步骤2: 轻量清洗")
            print("-"*70)
            
            cleaner = DataCleaner()
            cleaned_texts, valid_indices = cleaner.filter_batch(texts)
            print(f"✓ 清洗完成: {len(cleaned_texts)} / {len(texts)} ({len(cleaned_texts)/len(texts):.1%})")
        else:
            print("\n步骤2: 跳过清洗")
            cleaned_texts = texts
            valid_indices = list(range(len(texts)))
        
        # 步骤3: 去重并统计频率
        print("\n步骤3: 去重并统计频率")
        print("-"*70)
        
        from collections import Counter
        text_freq = Counter(cleaned_texts)
        
        df_dedup = pd.DataFrame({
            'text': cleaned_texts,
            'original_index': valid_indices
        })
        unique_df = df_dedup.drop_duplicates(subset=['text'], keep='first').copy()
        unique_df['frequency'] = unique_df['text'].map(text_freq)
        
        unique_texts = unique_df['text'].tolist()
        unique_indices = unique_df['original_index'].tolist()
        frequencies = unique_df['frequency'].tolist()
        
        freq_array = np.array(frequencies)
        print(f"✓ 去重完成: {len(unique_texts)} 条唯一文本")
        print(f"  频率统计: min={freq_array.min()}, max={freq_array.max()}, mean={freq_array.mean():.1f}")
        
        # 步骤4: 生成embeddings
        print("\n步骤4: 生成文本embeddings")
        print("-"*70)
        
        embedder = TextEmbedder(model_name=self.embedding_model)
        embeddings = embedder.embed_batch(unique_texts, batch_size=64)
        print(f"✓ Embeddings生成完成: {embeddings.shape}")
        
        # 步骤5: 聚类
        print("\n步骤5: K-means聚类")
        print("-"*70)
        
        sampler = DiversitySampler(n_clusters=self.n_clusters)
        labels = sampler.fit_predict(embeddings)
        
        # 步骤6: 分层采样（3倍过采样，留给质量筛选）
        print("\n步骤6: 分层采样（保证多样性）")
        print("-"*70)
        
        n_intermediate = min(self.n_samples * 3, len(unique_texts))
        
        sampled_texts, sampled_embeddings, sampled_indices, sampling_stats = sampler.stratified_sample(
            texts=unique_texts,
            embeddings=embeddings,
            labels=labels,
            n_samples=n_intermediate,
            strategy="hybrid",
            min_per_cluster=max(5, self.n_samples // self.n_clusters // 5),
            frequencies=frequencies
        )
        
        print(f"✓ 中间采样: {len(sampled_texts)} 条")
        print(f"  簇覆盖率: {sampling_stats['coverage_rate']:.1%}")
        
        # 提取采样数据的频率和标签
        sampled_frequencies = [frequencies[i] for i in sampled_indices]
        sampled_labels = labels[sampled_indices]
        
        # 步骤7: 质量评分
        print("\n步骤7: 质量评分与筛选")
        print("-"*70)
        
        scorer = QualityScorer()
        scores = scorer.score_batch(sampled_texts)
        
        # 簇平衡筛选（确保每个簇都有代表）
        unique_sampled_labels = np.unique(sampled_labels)
        final_indices = []
        
        # 计算每个簇的目标数量
        for cluster_id in unique_sampled_labels:
            cluster_mask = (sampled_labels == cluster_id)
            cluster_sample_indices = np.where(cluster_mask)[0]
            
            # 按比例分配，但至少3条
            target = max(3, int(self.n_samples * len(cluster_sample_indices) / len(sampled_texts)))
            
            # 按质量排序，取Top-K
            cluster_scores = scores[cluster_sample_indices]
            sorted_indices = cluster_sample_indices[np.argsort(cluster_scores)[::-1]]
            
            n_take = min(target, len(sorted_indices))
            final_indices.extend(sorted_indices[:n_take])
        
        # 如果不足，补充高质量样本
        if len(final_indices) < self.n_samples:
            remaining = self.n_samples - len(final_indices)
            all_indices = set(range(len(sampled_texts)))
            remaining_indices = list(all_indices - set(final_indices))
            remaining_scores = scores[remaining_indices]
            top_remaining = [remaining_indices[i] for i in np.argsort(remaining_scores)[::-1][:remaining]]
            final_indices.extend(top_remaining)
        
        # 截断到目标数量
        final_indices = final_indices[:self.n_samples]
        
        # 提取最终结果
        final_texts = [sampled_texts[i] for i in final_indices]
        final_scores = scores[final_indices]
        final_frequencies = [sampled_frequencies[i] for i in final_indices]
        final_labels = sampled_labels[final_indices]
        final_original_indices = [unique_indices[sampled_indices[i]] for i in final_indices]
        
        print(f"✓ 最终筛选: {len(final_texts)} 条")
        print(f"  质量得分: {np.min(final_scores):.3f} - {np.max(final_scores):.3f} (mean: {np.mean(final_scores):.3f})")
        print(f"  簇覆盖: {len(np.unique(final_labels))}/{len(unique_sampled_labels)} = {len(np.unique(final_labels))/len(unique_sampled_labels):.1%}")
        
        # 步骤8: 保存结果
        print("\n步骤8: 保存结果")
        print("-"*70)
        
        output_df = pd.DataFrame({
            'text': final_texts,
            'frequency': final_frequencies,
            'quality_score': final_scores,
            'cluster_id': final_labels,
            'original_index': final_original_indices,
            'importance': ['高频' if f >= 10 else '中频' if f >= 2 else '低频' for f in final_frequencies],
            'label': ''
        })
        
        # 创建输出目录
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        
        print(f"✓ 结果已保存: {self.output_file}")
        
        # 打印统计
        print("\n" + "="*70)
        print("采样完成统计")
        print("="*70)
        print(f"\n输入数据: {len(texts)} 条")
        print(f"去重后: {len(unique_texts)} 条")
        print(f"最终输出: {len(final_texts)} 条")
        print(f"\n质量分布:")
        print(f"  优秀(>=0.7): {np.sum(final_scores >= 0.7)} 条")
        print(f"  良好(0.5-0.7): {np.sum((final_scores >= 0.5) & (final_scores < 0.7))} 条")
        print(f"  一般(<0.5): {np.sum(final_scores < 0.5)} 条")
        print(f"\n频率分布:")
        print(f"  高频(>=10): {np.sum(np.array(final_frequencies) >= 10)} 条")
        print(f"  中频(2-9): {np.sum((np.array(final_frequencies) >= 2) & (np.array(final_frequencies) < 10))} 条")
        print(f"  低频(1): {np.sum(np.array(final_frequencies) == 1)} 条")
        
        print("\n✓ 二次采样完成！可以开始标注了。")


def main():
    parser = argparse.ArgumentParser(
        description="二次采样 - 从已清洗数据中精选高质量样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法：
  # 从4万条清洗数据中采样1万条
  python secondary_sampling.py --input cleaned_40k.csv --output final_10k.csv --n_samples 10000 --n_clusters 300
  
  # 跳过轻量清洗（如果数据已经很干净）
  python secondary_sampling.py --input cleaned_40k.csv --output final_10k.csv --n_samples 10000 --no-clean
        """
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件（已清洗的数据）"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=10000,
        help="目标采样数量（默认: 10000）"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=300,
        help="聚类数量（默认: 300）"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="moka-ai/m3e-base",
        help="Embedding模型"
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="跳过轻量清洗（如果数据已经很干净）"
    )
    
    args = parser.parse_args()
    
    sampler = SecondarySampler(
        input_file=args.input,
        output_file=args.output,
        n_samples=args.n_samples,
        n_clusters=args.n_clusters,
        embedding_model=args.embedding_model,
        enable_light_cleaning=not args.no_clean
    )
    
    sampler.run()


if __name__ == "__main__":
    main()









