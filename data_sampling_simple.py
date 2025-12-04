"""
简化版数据采样Pipeline - 不依赖向量数据库

适用场景:
- 无法安装ChromaDB
- 数据量较小（<100万）
- 快速测试

流程:
1. 数据清洗
2. 精确去重
3. TF-IDF特征提取（替代深度学习嵌入）
4. K-means聚类
5. 质量评分与采样
"""

import pandas as pd
import numpy as np
import re
from typing import List, Tuple
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


# ===============================================
# 数据清洗模块（与完整版相同）
# ===============================================
class DataCleaner:
    """数据清洗器"""
    
    def __init__(self):
        self.invalid_patterns = [
            r'^[\s\W]*$',
            r'.*[\x00-\x1F].*',
            r'.*[\uFFFD].*',
        ]
        
        self.spam_keywords = [
            '广告', '推广', '加微信', 'VX', '扫码', '点击链接',
            'http://', 'https://', 'www.', '.com', '.cn',
            '￥', '$$$', '免费领取', '限时优惠'
        ]
    
    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def is_valid(self, text: str) -> bool:
        """检查文本是否有效"""
        # 长度检查
        if not (3 <= len(text.strip()) <= 500):
            return False
        
        # 格式检查
        for pattern in self.invalid_patterns:
            if re.match(pattern, text):
                return False
        
        # 垃圾信息检查
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in self.spam_keywords):
            return False
        
        return True
    
    def filter_batch(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """批量过滤"""
        cleaned_texts = []
        valid_indices = []
        
        for idx, text in enumerate(tqdm(texts, desc="数据清洗")):
            if pd.isna(text) or not isinstance(text, str):
                continue
            
            text = self.clean_text(text)
            
            if self.is_valid(text):
                cleaned_texts.append(text)
                valid_indices.append(idx)
        
        return cleaned_texts, valid_indices


# ===============================================
# TF-IDF向量化模块（替代深度学习嵌入）
# ===============================================
class TfidfVectorizer:
    """TF-IDF向量化器 - 轻量级替代方案"""
    
    def __init__(self, max_features: int = 5000):
        """
        初始化TF-IDF向量化器
        
        参数:
        - max_features: 特征维度（词汇表大小）
        """
        from sklearn.feature_extraction.text import TfidfVectorizer as SklearnTfidf
        import jieba
        
        print(f"初始化TF-IDF向量化器 (max_features={max_features})")
        
        # 自定义分词函数
        def tokenizer(text):
            return list(jieba.cut(text))
        
        self.vectorizer = SklearnTfidf(
            max_features=max_features,
            tokenizer=tokenizer,
            lowercase=False,
            min_df=2,  # 至少出现2次
            max_df=0.8,  # 最多在80%文档中出现
        )
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        训练并转换文本
        
        返回:
        - vectors: (N, max_features) 稀疏矩阵
        """
        print("生成TF-IDF向量...")
        vectors = self.vectorizer.fit_transform(texts)
        
        # 转为dense（如果数据量不大）
        if vectors.shape[0] < 500000:
            vectors = vectors.toarray()
        
        print(f"✓ 向量化完成: shape={vectors.shape}")
        return vectors


# ===============================================
# 聚类采样模块
# ===============================================
class DiversitySampler:
    """多样性采样器"""
    
    def __init__(self, n_clusters: int = 100, random_state: int = 42):
        from sklearn.cluster import MiniBatchKMeans
        
        self.n_clusters = n_clusters
        self.kmeans = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            batch_size=1000,
            verbose=0
        )
    
    def fit_predict(self, vectors: np.ndarray) -> np.ndarray:
        """聚类"""
        print(f"执行K-means聚类 (k={self.n_clusters})...")
        labels = self.kmeans.fit_predict(vectors)
        
        unique, counts = np.unique(labels, return_counts=True)
        print(f"✓ 聚类完成: {len(unique)}个簇")
        print(f"   样本分布: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
        
        return labels
    
    def stratified_sample(
        self,
        texts: List[str],
        labels: np.ndarray,
        n_samples: int,
        strategy: str = "balanced"
    ) -> Tuple[List[str], np.ndarray]:
        """分层采样"""
        print(f"分层采样 (目标: {n_samples}条, 策略: {strategy})...")
        
        sampled_indices = []
        
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            if strategy == "balanced":
                n_cluster_samples = n_samples // self.n_clusters
            else:
                n_cluster_samples = int(n_samples * len(cluster_indices) / len(texts))
            
            n_cluster_samples = min(n_cluster_samples, len(cluster_indices))
            
            if n_cluster_samples > 0:
                selected = np.random.choice(
                    cluster_indices,
                    size=n_cluster_samples,
                    replace=False
                )
                sampled_indices.extend(selected)
        
        # 补充到目标数量
        if len(sampled_indices) < n_samples:
            remaining = n_samples - len(sampled_indices)
            all_indices = set(range(len(texts)))
            available = list(all_indices - set(sampled_indices))
            
            if len(available) >= remaining:
                additional = np.random.choice(available, size=remaining, replace=False)
                sampled_indices.extend(additional)
        
        sampled_indices = np.array(sampled_indices[:n_samples])
        
        print(f"✓ 采样完成: {len(sampled_indices)} 条")
        
        sampled_texts = [texts[i] for i in sampled_indices]
        return sampled_texts, sampled_indices


# ===============================================
# 质量评分模块
# ===============================================
class QualityScorer:
    """质量评分器"""
    
    def __init__(self):
        self.domain_keywords = [
            '寿险', '终身寿险', '定期寿险', '保险', '保障',
            '理赔', '受益人', '保额', '保费', '投保',
            '保单', '身故', '全残', '责任', '现金价值'
        ]
    
    def score_single(self, text: str) -> float:
        """单个文本评分"""
        # 1. 信息密度
        word_count = len(text)
        if 10 <= word_count <= 200:
            length_score = 1.0
        elif word_count < 10:
            length_score = word_count / 10
        else:
            length_score = max(0.5, 1 - (word_count - 200) / 300)
        
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        char_ratio = chinese_chars / max(len(text), 1)
        density = (length_score + char_ratio) / 2
        
        # 2. 语法完整性
        grammar = 0.0
        if re.search(r'[，。！？、；：]', text):
            grammar += 0.3
        if text.endswith('？') or text.endswith('?'):
            grammar += 0.3
        if not re.search(r'[^\u4e00-\u9fffa-zA-Z0-9]{3,}', text):
            grammar += 0.4
        grammar = min(grammar, 1.0)
        
        # 3. 领域相关性
        keyword_count = sum(1 for kw in self.domain_keywords if kw in text)
        if keyword_count == 0:
            relevance = 0.0
        elif keyword_count == 1:
            relevance = 0.5
        elif keyword_count == 2:
            relevance = 0.75
        else:
            relevance = 1.0
        
        # 综合得分
        final_score = 0.4 * relevance + 0.3 * density + 0.3 * grammar
        return final_score
    
    def score_batch(self, texts: List[str]) -> np.ndarray:
        """批量评分"""
        print("计算质量得分...")
        scores = [self.score_single(text) for text in tqdm(texts, desc="质量评分")]
        scores = np.array(scores)
        
        print(f"✓ 得分统计: min={scores.min():.2f}, max={scores.max():.2f}, mean={scores.mean():.2f}")
        return scores


# ===============================================
# 简化Pipeline
# ===============================================
class SimpleSamplingPipeline:
    """简化版采样Pipeline"""
    
    def __init__(
        self,
        input_file: str,
        output_file: str = "data/sampled_for_annotation.csv",
        n_target_samples: int = 10000,
        n_clusters: int = 200
    ):
        self.input_file = input_file
        self.output_file = output_file
        self.n_target_samples = n_target_samples
        self.n_clusters = n_clusters
        
        self.cleaner = DataCleaner()
        self.vectorizer = TfidfVectorizer()
        self.sampler = DiversitySampler(n_clusters=n_clusters)
        self.scorer = QualityScorer()
    
    def run(self):
        """运行Pipeline"""
        print("=" * 70)
        print("简化版数据采样Pipeline")
        print(f"输入: {self.input_file}")
        print(f"目标: {self.n_target_samples} 条")
        print("=" * 70)
        
        # 1. 加载数据
        print("\n步骤1: 加载数据")
        
        # 尝试多种编码加载CSV
        encodings_to_try = ['utf-8', 'gbk', 'gb18030', 'gb2312', 'latin1', 'iso-8859-1']
        df = None
        
        for encoding in encodings_to_try:
            try:
                print(f"尝试编码: {encoding}")
                df = pd.read_csv(self.input_file, encoding=encoding)
                print(f"✓ 加载成功（编码: {encoding}）")
                break
            except UnicodeDecodeError:
                print(f"  编码 {encoding} 失败")
                continue
            except Exception as e:
                print(f"  编码 {encoding} 出错: {str(e)[:60]}")
                continue
        
        if df is None:
            # 最后手段：忽略编码错误
            print("⚠️ 所有编码失败，尝试忽略编码错误...")
            df = pd.read_csv(self.input_file, encoding='utf-8', encoding_errors='ignore')
        
        # 查找文本列
        text_column = None
        for col in ['text', 'query', 'question', 'content', 'message']:
            if col in df.columns:
                text_column = col
                break
        
        if text_column is None:
            print(f"错误: 未找到文本列")
            return
        
        texts = df[text_column].tolist()
        print(f"✓ 加载完成: {len(texts)} 条")
        
        # 2. 数据清洗
        print("\n步骤2: 数据清洗")
        cleaned_texts, valid_indices = self.cleaner.filter_batch(texts)
        print(f"✓ 清洗完成: {len(cleaned_texts)} 条")
        
        # 3. 精确去重
        print("\n步骤3: 精确去重")
        unique_df = pd.DataFrame({'text': cleaned_texts, 'idx': valid_indices})
        unique_df = unique_df.drop_duplicates(subset=['text'])
        cleaned_texts = unique_df['text'].tolist()
        valid_indices = unique_df['idx'].tolist()
        print(f"✓ 去重完成: {len(cleaned_texts)} 条")
        
        # 如果数据量太大，先随机采样
        if len(cleaned_texts) > 500000:
            print(f"数据量较大，先随机采样到50万条...")
            sample_idx = np.random.choice(len(cleaned_texts), size=500000, replace=False)
            cleaned_texts = [cleaned_texts[i] for i in sample_idx]
            valid_indices = [valid_indices[i] for i in sample_idx]
        
        # 4. TF-IDF向量化
        print("\n步骤4: TF-IDF向量化")
        vectors = self.vectorizer.fit_transform(cleaned_texts)
        
        # 5. K-means聚类 + 采样
        print("\n步骤5: K-means聚类 + 多样性采样")
        n_intermediate = min(self.n_target_samples * 3, len(cleaned_texts))
        
        labels = self.sampler.fit_predict(vectors)
        sampled_texts, sampled_indices = self.sampler.stratified_sample(
            texts=cleaned_texts,
            labels=labels,
            n_samples=n_intermediate,
            strategy="balanced"
        )
        
        # 6. 质量评分
        print("\n步骤6: 质量评分与Top-K筛选")
        scores = self.scorer.score_batch(sampled_texts)
        
        top_k_idx = np.argsort(scores)[::-1][:self.n_target_samples]
        final_texts = [sampled_texts[i] for i in top_k_idx]
        final_scores = scores[top_k_idx]
        
        print(f"✓ 最终筛选: {len(final_texts)} 条")
        
        # 7. 保存结果
        print("\n步骤7: 保存结果")
        from pathlib import Path
        
        output_df = pd.DataFrame({
            'text': final_texts,
            'quality_score': final_scores,
            'label': ''  # 待标注
        })
        
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(self.output_file, index=False, encoding='utf-8-sig')
        
        print(f"✓ 数据已保存到: {self.output_file}")
        print("\n" + "=" * 70)
        print("✓ Pipeline完成！")
        print("=" * 70)


# ===============================================
# 命令行入口
# ===============================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="简化版数据采样Pipeline")
    parser.add_argument("--input", type=str, required=True, help="输入文件")
    parser.add_argument("--output", type=str, default="data/sampled_for_annotation.csv", help="输出文件")
    parser.add_argument("--n_samples", type=int, default=10000, help="目标采样数量")
    parser.add_argument("--n_clusters", type=int, default=200, help="聚类数量")
    
    args = parser.parse_args()
    
    pipeline = SimpleSamplingPipeline(
        input_file=args.input,
        output_file=args.output,
        n_target_samples=args.n_samples,
        n_clusters=args.n_clusters
    )
    
    pipeline.run()

