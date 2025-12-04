"""
聚类方法对比实验脚本

用途：快速对比K-means, Hybrid, HDBSCAN等方法在您数据上的效果

使用方法：
    python compare_clustering_methods.py --input data/cleaned_460k.csv --n_samples 5000

输出：
    - 各方法的簇覆盖率、标准化熵、轮廓系数
    - 可视化对比图表
    - 推荐使用哪种方法
"""

import pandas as pd
import numpy as np
import argparse
import time
from pathlib import Path
from typing import Dict, List
import json

# 导入必要的模块
import sys
sys.path.append(str(Path(__file__).parent))


def load_and_prepare_data(
    input_file: str,
    n_samples: int = 5000,
    embedding_model: str = "moka-ai/m3e-base"
) -> tuple:
    """
    加载并准备测试数据
    
    返回:
    - texts: 文本列表
    - embeddings: embedding矩阵
    """
    print(f"加载数据: {input_file}")
    
    # 加载数据
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file, nrows=n_samples)
    elif input_file.endswith('.json'):
        df = pd.read_json(input_file, lines=True, nrows=n_samples)
    else:
        raise ValueError(f"不支持的文件格式: {input_file}")
    
    # 找到文本列
    text_column = None
    for col in ['text', 'query', 'question', 'content', 'message']:
        if col in df.columns:
            text_column = col
            break
    
    if text_column is None:
        raise ValueError(f"未找到文本列，可用列: {df.columns.tolist()}")
    
    texts = df[text_column].tolist()
    print(f"✓ 加载了 {len(texts)} 条文本")
    
    # 生成embeddings
    print(f"生成embeddings (model={embedding_model})...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    print(f"✓ Embeddings shape: {embeddings.shape}")
    
    return texts, embeddings


def evaluate_clustering(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method_name: str
) -> Dict:
    """
    评估聚类质量
    
    指标：
    1. Silhouette Score (轮廓系数) - 越接近1越好
    2. Davies-Bouldin Index - 越小越好
    3. Calinski-Harabasz Score - 越大越好
    4. 簇大小分布
    5. Shannon熵（分布均匀性）
    """
    from sklearn.metrics import (
        silhouette_score,
        davies_bouldin_score,
        calinski_harabasz_score
    )
    
    print(f"\n评估 {method_name}...")
    
    results = {'method': method_name}
    
    # 过滤噪声点（如果有）
    valid_mask = labels != -1
    valid_embeddings = embeddings[valid_mask]
    valid_labels = labels[valid_mask]
    
    n_noise = np.sum(~valid_mask)
    n_clusters = len(np.unique(valid_labels))
    
    results['n_clusters'] = n_clusters
    results['n_noise'] = n_noise
    
    # 1. Silhouette Score（采样计算，加速）
    try:
        sample_size = min(1000, len(valid_embeddings))
        silhouette = silhouette_score(
            valid_embeddings,
            valid_labels,
            sample_size=sample_size
        )
        results['silhouette'] = float(silhouette)
    except Exception as e:
        print(f"  警告: 无法计算Silhouette Score: {e}")
        results['silhouette'] = 0.0
    
    # 2. Davies-Bouldin Index
    try:
        davies_bouldin = davies_bouldin_score(valid_embeddings, valid_labels)
        results['davies_bouldin'] = float(davies_bouldin)
    except Exception as e:
        print(f"  警告: 无法计算Davies-Bouldin: {e}")
        results['davies_bouldin'] = 999.0
    
    # 3. Calinski-Harabasz Score
    try:
        calinski = calinski_harabasz_score(valid_embeddings, valid_labels)
        results['calinski_harabasz'] = float(calinski)
    except Exception as e:
        print(f"  警告: 无法计算Calinski-Harabasz: {e}")
        results['calinski_harabasz'] = 0.0
    
    # 4. 簇大小分布
    unique, counts = np.unique(valid_labels, return_counts=True)
    results['cluster_size_min'] = int(counts.min())
    results['cluster_size_max'] = int(counts.max())
    results['cluster_size_mean'] = float(counts.mean())
    results['cluster_size_std'] = float(counts.std())
    
    # 5. Shannon熵（分布均匀性）
    cluster_probs = counts / counts.sum()
    shannon_entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
    max_entropy = np.log(len(counts))
    normalized_entropy = shannon_entropy / max_entropy if max_entropy > 0 else 0
    results['shannon_entropy'] = float(shannon_entropy)
    results['normalized_entropy'] = float(normalized_entropy)
    
    # 打印结果
    print(f"  簇数: {n_clusters}, 噪声点: {n_noise}")
    print(f"  轮廓系数: {results['silhouette']:.3f} (越接近1越好)")
    print(f"  Davies-Bouldin: {results['davies_bouldin']:.3f} (越小越好)")
    print(f"  Calinski-Harabasz: {results['calinski_harabasz']:.1f} (越大越好)")
    print(f"  标准化熵: {results['normalized_entropy']:.3f} (越接近1越好)")
    print(f"  簇大小: {results['cluster_size_min']}-{results['cluster_size_max']} (mean={results['cluster_size_mean']:.1f})")
    
    return results


def compare_methods(
    embeddings: np.ndarray,
    texts: List[str],
    n_clusters: int = 100,
    methods: List[str] = None
) -> Dict[str, Dict]:
    """
    对比多种聚类方法
    
    参数:
    - embeddings: embedding矩阵
    - texts: 文本列表
    - n_clusters: 目标簇数
    - methods: 要对比的方法列表
    
    返回:
    - results: {method_name: evaluation_results}
    """
    if methods is None:
        methods = ['kmeans', 'hybrid']  # 默认对比这两种
    
    print("=" * 70)
    print(f"聚类方法对比实验")
    print(f"数据量: {len(embeddings)}")
    print(f"目标簇数: {n_clusters}")
    print(f"对比方法: {', '.join(methods)}")
    print("=" * 70)
    
    results = {}
    
    for method in methods:
        print(f"\n{'='*70}")
        print(f"方法: {method.upper()}")
        print('='*70)
        
        # 记录时间
        start_time = time.time()
        
        try:
            # 导入并运行聚类
            if method == 'kmeans':
                from sklearn.cluster import MiniBatchKMeans
                clusterer = MiniBatchKMeans(
                    n_clusters=n_clusters,
                    random_state=42,
                    batch_size=1000
                )
                labels = clusterer.fit_predict(embeddings)
            
            elif method == 'hybrid':
                from improved_clustering_sampler import ImprovedDiversitySampler
                sampler = ImprovedDiversitySampler(
                    method='hybrid',
                    n_clusters=n_clusters
                )
                labels = sampler.fit_predict(embeddings)
            
            elif method == 'hdbscan':
                try:
                    import hdbscan
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=15,
                        min_samples=5,
                        metric='cosine',
                        cluster_selection_method='eom'
                    )
                    labels = clusterer.fit_predict(embeddings)
                except ImportError:
                    print("  ⚠️ 未安装hdbscan，跳过")
                    print("  安装: pip install hdbscan")
                    continue
            
            elif method == 'agglomerative':
                from sklearn.cluster import AgglomerativeClustering
                # 对大数据使用两阶段聚类
                if len(embeddings) > 10000:
                    print("  使用两阶段聚类（数据量大）")
                    # 阶段1: K-means粗聚类
                    from sklearn.cluster import MiniBatchKMeans
                    kmeans = MiniBatchKMeans(n_clusters=min(n_clusters*10, len(embeddings)//2))
                    coarse_labels = kmeans.fit_predict(embeddings)
                    # 阶段2: 对簇中心聚类
                    centers = kmeans.cluster_centers_
                    clusterer = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric='cosine',
                        linkage='average'
                    )
                    center_labels = clusterer.fit_predict(centers)
                    labels = center_labels[coarse_labels]
                else:
                    clusterer = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric='cosine',
                        linkage='average'
                    )
                    labels = clusterer.fit_predict(embeddings)
            
            else:
                print(f"  ⚠️ 未知方法: {method}")
                continue
            
            # 记录时间
            elapsed_time = time.time() - start_time
            
            # 评估
            result = evaluate_clustering(embeddings, labels, method)
            result['time'] = elapsed_time
            
            results[method] = result
            
            print(f"  ✓ 完成，耗时: {elapsed_time:.1f}秒")
        
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def print_comparison_summary(results: Dict[str, Dict]):
    """打印对比总结"""
    print("\n" + "=" * 70)
    print("对比总结")
    print("=" * 70)
    
    if len(results) == 0:
        print("没有成功的实验结果")
        return
    
    # 创建对比表
    print(f"\n{'方法':<15} {'簇数':<8} {'轮廓系数':<12} {'标准化熵':<12} {'时间(秒)':<10}")
    print("-" * 70)
    
    for method, result in results.items():
        print(f"{method:<15} "
              f"{result['n_clusters']:<8} "
              f"{result['silhouette']:<12.3f} "
              f"{result['normalized_entropy']:<12.3f} "
              f"{result['time']:<10.1f}")
    
    # 推荐
    print("\n" + "=" * 70)
    print("推荐")
    print("=" * 70)
    
    # 找出最佳方法
    best_quality = max(results.items(), key=lambda x: x[1]['silhouette'])
    best_diversity = max(results.items(), key=lambda x: x[1]['normalized_entropy'])
    fastest = min(results.items(), key=lambda x: x[1]['time'])
    
    print(f"\n最高质量（轮廓系数）: {best_quality[0]} ({best_quality[1]['silhouette']:.3f})")
    print(f"最高多样性（标准化熵）: {best_diversity[0]} ({best_diversity[1]['normalized_entropy']:.3f})")
    print(f"最快速度: {fastest[0]} ({fastest[1]['time']:.1f}秒)")
    
    # 综合推荐
    print("\n💡 综合推荐:")
    
    if 'hybrid' in results:
        hybrid_result = results['hybrid']
        print(f"  ✨ Hybrid方法 - 平衡速度和质量")
        print(f"     轮廓系数: {hybrid_result['silhouette']:.3f}")
        print(f"     标准化熵: {hybrid_result['normalized_entropy']:.3f}")
        print(f"     时间: {hybrid_result['time']:.1f}秒")
        
        if hybrid_result['silhouette'] > 0.25 and hybrid_result['normalized_entropy'] > 0.75:
            print(f"     状态: ✅ 质量和多样性都很好")
        else:
            print(f"     状态: ⚠️ 可能需要调整参数")
    
    if 'hdbscan' in results:
        hdbscan_result = results['hdbscan']
        print(f"\n  🎯 HDBSCAN - 追求极致精度")
        print(f"     轮廓系数: {hdbscan_result['silhouette']:.3f}")
        print(f"     标准化熵: {hdbscan_result['normalized_entropy']:.3f}")
        print(f"     时间: {hdbscan_result['time']:.1f}秒")
        
        if hdbscan_result['silhouette'] > best_quality[1]['silhouette'] * 1.05:
            print(f"     状态: ✅ 比K-means显著更好，建议使用")
        else:
            print(f"     状态: ⚠️ 提升有限，可根据时间成本决定")
    
    # 给出最终建议
    print("\n🎯 最终建议:")
    
    if 'kmeans' in results and 'hybrid' in results:
        kmeans_score = results['kmeans']['silhouette']
        hybrid_score = results['hybrid']['silhouette']
        improvement = (hybrid_score - kmeans_score) / kmeans_score * 100
        
        if improvement > 10:
            print(f"  ✅ Hybrid比K-means提升 {improvement:.1f}%，强烈建议升级")
        elif improvement > 5:
            print(f"  ✅ Hybrid比K-means提升 {improvement:.1f}%，建议升级")
        else:
            print(f"  ⚠️ Hybrid提升不明显（{improvement:.1f}%），继续使用K-means也可")
    
    print("\n  根据您的目标选择:")
    print("    - 速度优先 → K-means")
    print("    - 平衡方案 → Hybrid（推荐）")
    print("    - 质量优先 → HDBSCAN")


def save_results(results: Dict[str, Dict], output_file: str):
    """保存结果到JSON"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, indent=2, fp=f, ensure_ascii=False)
    
    print(f"\n✓ 结果已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="聚类方法对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件路径"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=5000,
        help="测试样本数（建议5000-10000）"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=100,
        help="目标簇数"
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="kmeans,hybrid",
        help="对比的方法，逗号分隔 (kmeans,hybrid,hdbscan,agglomerative)"
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="moka-ai/m3e-base",
        help="Embedding模型"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="clustering_comparison_results.json",
        help="结果输出文件"
    )
    
    args = parser.parse_args()
    
    # 解析方法列表
    methods = [m.strip() for m in args.methods.split(',')]
    
    # 加载数据
    texts, embeddings = load_and_prepare_data(
        args.input,
        args.n_samples,
        args.embedding_model
    )
    
    # 对比实验
    results = compare_methods(
        embeddings,
        texts,
        args.n_clusters,
        methods
    )
    
    # 打印总结
    print_comparison_summary(results)
    
    # 保存结果
    save_results(results, args.output)


if __name__ == "__main__":
    main()

