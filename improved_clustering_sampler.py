"""
改进的聚类采样器 - 针对意图分类优化

对比K-means，改进方法：
1. 使用HDBSCAN（基于密度）或AgglomerativeClustering（层次聚类）
2. 自适应确定簇数
3. 识别噪声点
4. 更适合不规则形状的簇

适用场景：
- 文本意图分类
- 需要高精度多样性采样
- 目标：98% F1
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class ImprovedDiversitySampler:
    """
    改进的多样性采样器
    
    支持多种聚类算法：
    1. MiniBatchKMeans - 快速，适合大数据
    2. HDBSCAN - 基于密度，自适应簇数，识别噪声
    3. AgglomerativeClustering - 层次聚类，更准确
    4. Hybrid - 混合方法（推荐）
    """
    
    def __init__(
        self,
        method: str = "hybrid",
        n_clusters: int = 300,
        random_state: int = 42,
        min_cluster_size: int = 15,  # HDBSCAN参数
        metric: str = "cosine"       # 距离度量
    ):
        """
        初始化改进的聚类采样器
        
        参数:
        - method: 聚类方法
            - 'kmeans': 标准K-means（快速）
            - 'hdbscan': 基于密度聚类（高质量，自适应K）
            - 'agglomerative': 层次聚类（中等质量，固定K）
            - 'hybrid': 混合方法（推荐，平衡速度和质量）
        - n_clusters: 目标簇数（kmeans/agglomerative）
        - random_state: 随机种子
        - min_cluster_size: HDBSCAN最小簇大小
        - metric: 距离度量（cosine更适合文本）
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.clusterer = None
        self.labels_ = None
        
        print(f"初始化聚类采样器: method={method}, n_clusters={n_clusters}")
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        聚类并预测每个样本的类别
        
        参数:
        - embeddings: (N, D) 嵌入矩阵（已归一化）
        
        返回:
        - labels: (N,) 类别标签
        """
        print(f"执行聚类 (method={self.method}, n_samples={len(embeddings)})...")
        
        if self.method == "kmeans":
            labels = self._kmeans_clustering(embeddings)
        elif self.method == "hdbscan":
            labels = self._hdbscan_clustering(embeddings)
        elif self.method == "agglomerative":
            labels = self._agglomerative_clustering(embeddings)
        elif self.method == "hybrid":
            labels = self._hybrid_clustering(embeddings)
        else:
            raise ValueError(f"未知的聚类方法: {self.method}")
        
        self.labels_ = labels
        
        # 统计聚类结果
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels)
        n_noise = np.sum(labels == -1) if -1 in labels else 0
        
        if n_noise > 0:
            # 排除噪声点统计
            valid_labels = labels[labels != -1]
            unique, counts = np.unique(valid_labels, return_counts=True)
            print(f"✓ 聚类完成: {n_clusters-1}个簇 + {n_noise}个噪声点")
        else:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"✓ 聚类完成: {n_clusters}个簇")
        
        print(f"  簇大小分布: min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")
        
        return labels
    
    def _kmeans_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """标准K-means聚类"""
        from sklearn.cluster import MiniBatchKMeans
        
        self.clusterer = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            batch_size=1000,
            verbose=0
        )
        
        labels = self.clusterer.fit_predict(embeddings)
        return labels
    
    def _hdbscan_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        HDBSCAN聚类（基于密度）
        
        优势：
        - 自动确定簇数
        - 识别噪声点
        - 支持不规则形状的簇
        - 对异常值鲁棒
        """
        try:
            import hdbscan
        except ImportError:
            print("警告: 未安装hdbscan，降级到K-means")
            print("安装: pip install hdbscan")
            return self._kmeans_clustering(embeddings)
        
        print(f"  使用HDBSCAN (min_cluster_size={self.min_cluster_size})")
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=5,
            metric=self.metric,
            cluster_selection_method='eom',  # Excess of Mass
            prediction_data=True
        )
        
        labels = self.clusterer.fit_predict(embeddings)
        
        # HDBSCAN返回-1表示噪声点
        # 将噪声点分配到最近的簇
        if np.sum(labels == -1) > 0:
            labels = self._assign_noise_to_nearest_cluster(embeddings, labels)
        
        return labels
    
    def _agglomerative_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        层次聚类
        
        优势：
        - 不假设簇形状
        - 考虑层次结构
        - 对于小规模数据很准确
        
        劣势：
        - 对大数据较慢（O(n²)）
        """
        from sklearn.cluster import AgglomerativeClustering
        
        # 对于大数据，先用K-means粗聚类，再用层次聚类精细化
        if len(embeddings) > 100000:
            print(f"  数据量大，使用两阶段聚类")
            # 第一阶段：K-means粗聚类到10倍簇数
            coarse_clusters = self.n_clusters * 10
            kmeans = MiniBatchKMeans(n_clusters=min(coarse_clusters, len(embeddings)//2))
            coarse_labels = kmeans.fit_predict(embeddings)
            
            # 第二阶段：对簇中心进行层次聚类
            centers = kmeans.cluster_centers_
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric='cosine',
                linkage='average'
            )
            center_labels = self.clusterer.fit_predict(centers)
            
            # 映射回原始数据
            labels = center_labels[coarse_labels]
        else:
            print(f"  使用标准层次聚类")
            self.clusterer = AgglomerativeClustering(
                n_clusters=self.n_clusters,
                metric='cosine',
                linkage='average'
            )
            labels = self.clusterer.fit_predict(embeddings)
        
        return labels
    
    def _hybrid_clustering(self, embeddings: np.ndarray) -> np.ndarray:
        """
        混合聚类方法（推荐）
        
        策略：
        1. 先用K-means快速粗聚类
        2. 识别每个簇的密度和大小
        3. 对大簇进一步细分
        4. 对小簇进行合并或标记
        
        平衡了速度和质量
        """
        print(f"  使用混合聚类策略")
        
        # 第一阶段：K-means粗聚类
        print(f"    阶段1: K-means粗聚类到 {self.n_clusters}簇")
        labels = self._kmeans_clustering(embeddings)
        
        # 第二阶段：分析簇质量
        print(f"    阶段2: 分析簇质量")
        cluster_stats = self._analyze_cluster_quality(embeddings, labels)
        
        # 第三阶段：优化簇划分
        print(f"    阶段3: 优化簇划分")
        labels = self._refine_clusters(embeddings, labels, cluster_stats)
        
        return labels
    
    def _analyze_cluster_quality(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, Dict]:
        """
        分析每个簇的质量
        
        指标：
        - size: 簇大小
        - density: 簇密度（平均内聚度）
        - diameter: 簇直径（最大距离）
        """
        cluster_stats = {}
        
        for cluster_id in np.unique(labels):
            cluster_mask = (labels == cluster_id)
            cluster_embeddings = embeddings[cluster_mask]
            
            # 计算簇中心
            center = cluster_embeddings.mean(axis=0)
            
            # 计算密度（平均与中心的距离）
            distances = np.linalg.norm(cluster_embeddings - center, axis=1)
            density = 1.0 / (distances.mean() + 1e-6)
            
            # 计算直径
            diameter = distances.max()
            
            cluster_stats[cluster_id] = {
                'size': len(cluster_embeddings),
                'density': density,
                'diameter': diameter,
                'center': center
            }
        
        return cluster_stats
    
    def _refine_clusters(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        cluster_stats: Dict[int, Dict]
    ) -> np.ndarray:
        """
        优化簇划分
        
        规则：
        1. 大簇且密度低 → 进一步细分
        2. 小簇且密度高 → 保持
        3. 小簇且密度低 → 可能是噪声，合并到最近的簇
        """
        refined_labels = labels.copy()
        
        # 计算阈值
        sizes = [stats['size'] for stats in cluster_stats.values()]
        densities = [stats['density'] for stats in cluster_stats.values()]
        
        size_threshold = np.percentile(sizes, 75)  # 大簇阈值
        density_threshold = np.percentile(densities, 25)  # 低密度阈值
        
        new_cluster_id = labels.max() + 1
        
        for cluster_id, stats in cluster_stats.items():
            cluster_mask = (labels == cluster_id)
            
            # 规则1：大簇且密度低 → 细分
            if stats['size'] > size_threshold and stats['density'] < density_threshold:
                cluster_embeddings = embeddings[cluster_mask]
                
                # 使用K-means将该簇分为2-3个子簇
                n_sub_clusters = min(3, stats['size'] // 50)
                if n_sub_clusters > 1:
                    from sklearn.cluster import KMeans
                    sub_kmeans = KMeans(n_clusters=n_sub_clusters, random_state=self.random_state)
                    sub_labels = sub_kmeans.fit_predict(cluster_embeddings)
                    
                    # 更新标签
                    cluster_indices = np.where(cluster_mask)[0]
                    for i, sub_label in enumerate(sub_labels):
                        if sub_label > 0:
                            refined_labels[cluster_indices[i]] = new_cluster_id + sub_label - 1
                    
                    new_cluster_id += n_sub_clusters - 1
        
        print(f"    优化后簇数: {len(np.unique(refined_labels))}")
        
        return refined_labels
    
    def _assign_noise_to_nearest_cluster(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> np.ndarray:
        """将噪声点分配到最近的簇"""
        noise_mask = (labels == -1)
        n_noise = noise_mask.sum()
        
        if n_noise == 0:
            return labels
        
        print(f"  处理 {n_noise} 个噪声点...")
        
        # 计算每个簇的中心
        unique_labels = np.unique(labels[~noise_mask])
        centers = []
        for label in unique_labels:
            cluster_mask = (labels == label)
            center = embeddings[cluster_mask].mean(axis=0)
            centers.append(center)
        centers = np.array(centers)
        
        # 将噪声点分配到最近的簇
        noise_embeddings = embeddings[noise_mask]
        distances = np.linalg.norm(
            noise_embeddings[:, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=2
        )
        nearest_clusters = unique_labels[np.argmin(distances, axis=1)]
        
        # 更新标签
        labels = labels.copy()
        labels[noise_mask] = nearest_clusters
        
        return labels
    
    def stratified_sample(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        labels: np.ndarray,
        n_samples: int = 10000,
        strategy: str = "hybrid",
        min_per_cluster: int = 5,
        max_per_cluster: int = None
    ) -> Tuple[List[str], np.ndarray, np.ndarray, Dict]:
        """
        分层采样（与原版相同，支持改进的聚类结果）
        """
        # 这里可以直接使用原来的stratified_sample实现
        # 因为它只依赖labels，不依赖具体聚类算法
        
        from data_sampling_pipeline import DiversitySampler
        
        # 临时创建一个DiversitySampler来使用其stratified_sample方法
        temp_sampler = DiversitySampler(n_clusters=len(np.unique(labels)))
        
        return temp_sampler.stratified_sample(
            texts=texts,
            embeddings=embeddings,
            labels=labels,
            n_samples=n_samples,
            strategy=strategy,
            min_per_cluster=min_per_cluster,
            max_per_cluster=max_per_cluster
        )


def compare_clustering_methods(
    embeddings: np.ndarray,
    texts: List[str] = None,
    n_samples: int = 1000,
    n_clusters: int = 100
):
    """
    比较不同聚类方法的效果
    
    用于评估哪种方法最适合您的数据
    """
    print("=" * 70)
    print("聚类方法比较")
    print("=" * 70)
    
    methods = ["kmeans", "agglomerative", "hybrid"]
    results = {}
    
    for method in methods:
        print(f"\n测试方法: {method}")
        print("-" * 70)
        
        sampler = ImprovedDiversitySampler(
            method=method,
            n_clusters=n_clusters
        )
        
        # 只用前n_samples个样本进行快速测试
        test_embeddings = embeddings[:n_samples]
        labels = sampler.fit_predict(test_embeddings)
        
        # 评估指标
        from sklearn.metrics import silhouette_score, davies_bouldin_score
        
        # Silhouette Score (轮廓系数，越接近1越好)
        try:
            silhouette = silhouette_score(test_embeddings, labels, sample_size=min(1000, n_samples))
        except:
            silhouette = 0.0
        
        # Davies-Bouldin Index (越小越好)
        try:
            davies_bouldin = davies_bouldin_score(test_embeddings, labels)
        except:
            davies_bouldin = 999.0
        
        # 簇大小分布
        unique, counts = np.unique(labels, return_counts=True)
        
        results[method] = {
            'n_clusters': len(unique),
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'cluster_size_std': counts.std(),
            'cluster_size_min': counts.min(),
            'cluster_size_max': counts.max(),
        }
        
        print(f"  簇数: {results[method]['n_clusters']}")
        print(f"  轮廓系数: {results[method]['silhouette']:.3f} (越接近1越好)")
        print(f"  Davies-Bouldin: {results[method]['davies_bouldin']:.3f} (越小越好)")
        print(f"  簇大小分布: {results[method]['cluster_size_min']}-{results[method]['cluster_size_max']}")
    
    # 推荐
    print("\n" + "=" * 70)
    print("推荐")
    print("=" * 70)
    
    best_silhouette = max(results.values(), key=lambda x: x['silhouette'])
    best_method = [k for k, v in results.items() if v == best_silhouette][0]
    
    print(f"根据轮廓系数，推荐使用: {best_method}")
    print(f"但建议根据实际任务选择：")
    print(f"  - kmeans: 速度最快，适合大数据")
    print(f"  - hybrid: 平衡速度和质量（推荐）")
    print(f"  - agglomerative: 质量最高，适合中小数据")
    
    return results


if __name__ == "__main__":
    # 使用示例
    print("""
使用示例：

# 1. 替换原有的DiversitySampler
from improved_clustering_sampler import ImprovedDiversitySampler

# 2. 创建改进的采样器
sampler = ImprovedDiversitySampler(
    method="hybrid",  # 或 "kmeans", "hdbscan", "agglomerative"
    n_clusters=300,
    min_cluster_size=15
)

# 3. 后续使用方式完全相同
labels = sampler.fit_predict(embeddings)
sampled_texts, sampled_embeddings, sampled_indices, stats = sampler.stratified_sample(...)

# 4. 比较不同方法（可选）
results = compare_clustering_methods(embeddings, texts, n_samples=5000, n_clusters=200)
""")

