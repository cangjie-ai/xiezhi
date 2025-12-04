"""
可视化采样结果

生成各种图表，分析采样质量
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def visualize_results(
    original_file: str,
    sampled_file: str,
    output_dir: str = "visualization"
):
    """
    可视化分析采样结果
    
    生成图表:
    1. 类别分布对比
    2. 质量得分分布
    3. 文本长度分布
    4. 关键词覆盖热力图
    5. 聚类可视化（如果有嵌入向量）
    """
    
    print("=" * 70)
    print("可视化分析采样结果")
    print("=" * 70)
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    original_df = pd.read_csv(original_file)
    sampled_df = pd.read_csv(sampled_file)
    
    # 合并真实标签（如果有）
    if 'true_label' in original_df.columns:
        sampled_df = sampled_df.merge(
            original_df[['text', 'true_label']],
            on='text',
            how='left'
        )
    
    # ========== 图表1: 类别分布对比 ==========
    print("\n生成图表1: 类别分布对比...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 原始分布
    if 'true_label' in original_df.columns:
        original_counts = original_df['true_label'].value_counts()
        ax1.pie(
            original_counts.values,
            labels=original_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2")
        )
        ax1.set_title(f'原始数据分布\n(总计: {len(original_df):,} 条)', fontsize=12, fontweight='bold')
    
    # 采样分布
    if 'true_label' in sampled_df.columns:
        sampled_counts = sampled_df['true_label'].value_counts()
        ax2.pie(
            sampled_counts.values,
            labels=sampled_counts.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=sns.color_palette("Set2")
        )
        ax2.set_title(f'采样数据分布\n(总计: {len(sampled_df):,} 条)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/1_class_distribution.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ 已保存: {output_dir}/1_class_distribution.png")
    plt.close()
    
    # ========== 图表2: 质量得分分布 ==========
    if 'quality_score' in sampled_df.columns:
        print("\n生成图表2: 质量得分分布...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1.hist(sampled_df['quality_score'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(sampled_df['quality_score'].mean(), color='red', linestyle='--', 
                    label=f'均值: {sampled_df["quality_score"].mean():.3f}')
        ax1.axvline(sampled_df['quality_score'].median(), color='green', linestyle='--',
                    label=f'中位数: {sampled_df["quality_score"].median():.3f}')
        ax1.set_xlabel('质量得分', fontsize=11)
        ax1.set_ylabel('频数', fontsize=11)
        ax1.set_title('质量得分分布', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 箱线图
        if 'true_label' in sampled_df.columns:
            sampled_df.boxplot(column='quality_score', by='true_label', ax=ax2)
            ax2.set_xlabel('类别', fontsize=11)
            ax2.set_ylabel('质量得分', fontsize=11)
            ax2.set_title('各类别质量得分对比', fontsize=12, fontweight='bold')
            plt.sca(ax2)
            plt.xticks(rotation=15)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/2_quality_score.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ 已保存: {output_dir}/2_quality_score.png")
        plt.close()
    
    # ========== 图表3: 文本长度分布 ==========
    print("\n生成图表3: 文本长度分布...")
    
    sampled_df['text_len'] = sampled_df['text'].str.len()
    original_df['text_len'] = original_df['text'].str.len()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 对比直方图
    ax1.hist(original_df['text_len'], bins=50, alpha=0.5, label='原始数据', color='gray')
    ax1.hist(sampled_df['text_len'], bins=50, alpha=0.7, label='采样数据', color='orange')
    ax1.set_xlabel('文本长度（字符）', fontsize=11)
    ax1.set_ylabel('频数', fontsize=11)
    ax1.set_title('文本长度分布对比', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 分类箱线图
    if 'true_label' in sampled_df.columns:
        sampled_df.boxplot(column='text_len', by='true_label', ax=ax2)
        ax2.set_xlabel('类别', fontsize=11)
        ax2.set_ylabel('文本长度（字符）', fontsize=11)
        ax2.set_title('各类别文本长度对比', fontsize=12, fontweight='bold')
        plt.sca(ax2)
        plt.xticks(rotation=15)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/3_text_length.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ 已保存: {output_dir}/3_text_length.png")
    plt.close()
    
    # ========== 图表4: 关键词覆盖分析 ==========
    print("\n生成图表4: 关键词覆盖分析...")
    
    keywords = ['寿险', '终身', '定期', '保额', '保费', '理赔', '受益人', '投保', '保单', '现金价值']
    
    keyword_stats = []
    for kw in keywords:
        original_count = original_df['text'].str.contains(kw, na=False).sum()
        sampled_count = sampled_df['text'].str.contains(kw, na=False).sum()
        
        keyword_stats.append({
            '关键词': kw,
            '原始覆盖率': original_count / len(original_df) * 100,
            '采样覆盖率': sampled_count / len(sampled_df) * 100
        })
    
    keyword_df = pd.DataFrame(keyword_stats)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(keywords))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, keyword_df['原始覆盖率'], width, label='原始数据', alpha=0.7, color='gray')
    bars2 = ax.bar(x + width/2, keyword_df['采样覆盖率'], width, label='采样数据', alpha=0.7, color='orange')
    
    ax.set_xlabel('关键词', fontsize=11)
    ax.set_ylabel('覆盖率 (%)', fontsize=11)
    ax.set_title('关键词覆盖率对比', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(keywords, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/4_keyword_coverage.png", dpi=150, bbox_inches='tight')
    print(f"  ✓ 已保存: {output_dir}/4_keyword_coverage.png")
    plt.close()
    
    # ========== 图表5: 得分-长度关系 ==========
    if 'quality_score' in sampled_df.columns:
        print("\n生成图表5: 质量得分与文本长度关系...")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'true_label' in sampled_df.columns:
            for label in sampled_df['true_label'].unique():
                subset = sampled_df[sampled_df['true_label'] == label]
                ax.scatter(subset['text_len'], subset['quality_score'], 
                          alpha=0.5, s=30, label=label)
        else:
            ax.scatter(sampled_df['text_len'], sampled_df['quality_score'],
                      alpha=0.5, s=30, color='blue')
        
        ax.set_xlabel('文本长度（字符）', fontsize=11)
        ax.set_ylabel('质量得分', fontsize=11)
        ax.set_title('质量得分 vs 文本长度', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/5_score_vs_length.png", dpi=150, bbox_inches='tight')
        print(f"  ✓ 已保存: {output_dir}/5_score_vs_length.png")
        plt.close()
    
    # ========== 生成汇总报告 ==========
    print("\n生成汇总报告...")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("采样结果可视化分析报告")
    report_lines.append("=" * 70)
    report_lines.append(f"\n数据概览:")
    report_lines.append(f"  原始数据量: {len(original_df):,}")
    report_lines.append(f"  采样数据量: {len(sampled_df):,}")
    report_lines.append(f"  采样比例: {len(sampled_df)/len(original_df)*100:.2f}%")
    
    if 'quality_score' in sampled_df.columns:
        report_lines.append(f"\n质量统计:")
        report_lines.append(f"  平均得分: {sampled_df['quality_score'].mean():.3f}")
        report_lines.append(f"  中位数: {sampled_df['quality_score'].median():.3f}")
        report_lines.append(f"  标准差: {sampled_df['quality_score'].std():.3f}")
    
    report_lines.append(f"\n文本长度:")
    report_lines.append(f"  平均长度: {sampled_df['text_len'].mean():.1f} 字符")
    report_lines.append(f"  中位数: {sampled_df['text_len'].median():.1f} 字符")
    report_lines.append(f"  范围: {sampled_df['text_len'].min()}-{sampled_df['text_len'].max()} 字符")
    
    report_lines.append(f"\n生成的图表:")
    report_lines.append(f"  1. {output_dir}/1_class_distribution.png - 类别分布对比")
    if 'quality_score' in sampled_df.columns:
        report_lines.append(f"  2. {output_dir}/2_quality_score.png - 质量得分分布")
    report_lines.append(f"  3. {output_dir}/3_text_length.png - 文本长度分布")
    report_lines.append(f"  4. {output_dir}/4_keyword_coverage.png - 关键词覆盖分析")
    if 'quality_score' in sampled_df.columns:
        report_lines.append(f"  5. {output_dir}/5_score_vs_length.png - 得分-长度关系")
    
    report_lines.append("\n" + "=" * 70)
    
    report_text = "\n".join(report_lines)
    
    # 保存报告
    with open(f"{output_dir}/visualization_report.txt", 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n✓ 报告已保存: {output_dir}/visualization_report.txt")
    
    print("\n" + "=" * 70)
    print("✓ 可视化分析完成！")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="可视化采样结果")
    parser.add_argument("--original", type=str, required=True, help="原始数据文件")
    parser.add_argument("--sampled", type=str, required=True, help="采样结果文件")
    parser.add_argument("--output_dir", type=str, default="visualization", help="输出目录")
    
    args = parser.parse_args()
    
    visualize_results(
        original_file=args.original,
        sampled_file=args.sampled,
        output_dir=args.output_dir
    )

