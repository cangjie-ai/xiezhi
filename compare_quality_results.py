"""
对比BERT和LLM的数据质量检查结果

功能：
1. 加载BERT和LLM的质量检查结果
2. 对比两个模型检测出的问题样本
3. 找出共同标记的高置信度问题样本（更可靠）
4. 分析两个模型的一致性
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# ============================================
# 配置参数
# ============================================
BERT_REPORT = './cleanlab_results/data_quality_report.csv'
LLM_REPORT = './cleanlab_results_llm/data_quality_report_llm.csv'
OUTPUT_DIR = './quality_comparison'

print("=" * 70)
print("对比BERT和LLM的数据质量检查结果")
print("=" * 70)

# ============================================
# 第1步：加载结果
# ============================================
print("\n第1步：加载检查结果...")

try:
    df_bert = pd.read_csv(BERT_REPORT)
    print(f"✓ BERT结果: {len(df_bert)} 条样本")
except FileNotFoundError:
    print(f"✗ 找不到BERT结果文件: {BERT_REPORT}")
    print("请先运行 check_data_quality.py")
    exit(1)

try:
    df_llm = pd.read_csv(LLM_REPORT)
    print(f"✓ LLM结果: {len(df_llm)} 条样本")
except FileNotFoundError:
    print(f"✗ 找不到LLM结果文件: {LLM_REPORT}")
    print("请先运行 check_data_quality_llm.py")
    exit(1)

# 确保数据对齐
assert len(df_bert) == len(df_llm), "BERT和LLM结果样本数量不一致！"

# ============================================
# 第2步：对比分析
# ============================================
print("\n" + "=" * 70)
print("第2步：对比分析...")

# 合并数据
df_compare = pd.DataFrame({
    'text': df_bert['text'],
    'label': df_bert['label'],
    'bert_quality': df_bert['quality_score'],
    'llm_quality': df_llm['quality_score'],
    'bert_is_issue': df_bert['is_issue'],
    'llm_is_issue': df_llm['is_issue'],
    'bert_pred': df_bert['pred_label'],
    'llm_pred': df_llm['pred_label'],
})

# 计算一致性指标
both_issue = (df_compare['bert_is_issue'] & df_compare['llm_is_issue']).sum()
only_bert = (df_compare['bert_is_issue'] & ~df_compare['llm_is_issue']).sum()
only_llm = (~df_compare['bert_is_issue'] & df_compare['llm_is_issue']).sum()
both_ok = (~df_compare['bert_is_issue'] & ~df_compare['llm_is_issue']).sum()

print("\n问题样本检测对比:")
print(f"  两者都标记为问题: {both_issue} 条")
print(f"  仅BERT标记为问题: {only_bert} 条")
print(f"  仅LLM标记为问题: {only_llm} 条")
print(f"  两者都认为正常: {both_ok} 条")

# 计算一致性比例
agreement = (both_issue + both_ok) / len(df_compare) * 100
print(f"\n一致性: {agreement:.2f}%")

# 计算质量分数的相关性
corr_pearson, p_pearson = pearsonr(df_compare['bert_quality'], df_compare['llm_quality'])
corr_spearman, p_spearman = spearmanr(df_compare['bert_quality'], df_compare['llm_quality'])

print(f"\n质量分数相关性:")
print(f"  Pearson相关系数: {corr_pearson:.4f} (p={p_pearson:.4e})")
print(f"  Spearman相关系数: {corr_spearman:.4f} (p={p_spearman:.4e})")

# ============================================
# 第3步：找出高置信度问题样本
# ============================================
print("\n" + "=" * 70)
print("第3步：找出高置信度问题样本...")

# 共同标记的问题样本（更可靠）
df_compare['both_issue'] = df_compare['bert_is_issue'] & df_compare['llm_is_issue']
df_compare['min_quality'] = df_compare[['bert_quality', 'llm_quality']].min(axis=1)

high_confidence_issues = df_compare[df_compare['both_issue']].sort_values('min_quality')

print(f"\n两个模型共同标记的问题样本: {len(high_confidence_issues)} 条")
print("这些样本更可能存在标注错误，建议优先复核！")

# ============================================
# 第4步：保存结果
# ============================================
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存完整对比
output_file = f"{OUTPUT_DIR}/comparison_full.csv"
df_compare.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n完整对比已保存: {output_file}")

# 保存高置信度问题样本
high_conf_file = f"{OUTPUT_DIR}/high_confidence_issues.csv"
high_confidence_issues.to_csv(high_conf_file, index=False, encoding='utf-8-sig')
print(f"高置信度问题样本已保存: {high_conf_file}")

# 保存仅被一个模型标记的样本（需要进一步审查）
only_bert_df = df_compare[df_compare['bert_is_issue'] & ~df_compare['llm_is_issue']]
only_llm_df = df_compare[~df_compare['bert_is_issue'] & df_compare['llm_is_issue']]

only_bert_file = f"{OUTPUT_DIR}/only_bert_issues.csv"
only_llm_file = f"{OUTPUT_DIR}/only_llm_issues.csv"

only_bert_df.to_csv(only_bert_file, index=False, encoding='utf-8-sig')
only_llm_df.to_csv(only_llm_file, index=False, encoding='utf-8-sig')

print(f"仅BERT标记的问题已保存: {only_bert_file}")
print(f"仅LLM标记的问题已保存: {only_llm_file}")

# ============================================
# 第5步：可视化分析
# ============================================
print("\n" + "=" * 70)
print("第5步：生成可视化分析...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 图1: 质量分数散点图
ax1 = axes[0, 0]
colors = ['red' if both else 'gray' for both in df_compare['both_issue']]
ax1.scatter(df_compare['bert_quality'], df_compare['llm_quality'], 
           c=colors, alpha=0.5, s=10)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('BERT质量分数')
ax1.set_ylabel('LLM质量分数')
ax1.set_title(f'质量分数对比\n(红色=两者都标记为问题)')
ax1.grid(True, alpha=0.3)

# 图2: 质量分数分布对比
ax2 = axes[0, 1]
ax2.hist(df_compare['bert_quality'], bins=50, alpha=0.5, label='BERT', color='blue')
ax2.hist(df_compare['llm_quality'], bins=50, alpha=0.5, label='LLM', color='orange')
ax2.set_xlabel('质量分数')
ax2.set_ylabel('样本数量')
ax2.set_title('质量分数分布对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: 一致性矩阵
ax3 = axes[1, 0]
confusion_data = np.array([
    [both_ok, only_llm],
    [only_bert, both_issue]
])
im = ax3.imshow(confusion_data, cmap='Blues')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['LLM认为OK', 'LLM认为问题'])
ax3.set_yticklabels(['BERT认为OK', 'BERT认为问题'])
ax3.set_title('问题检测一致性矩阵')

# 添加数值标注
for i in range(2):
    for j in range(2):
        text = ax3.text(j, i, confusion_data[i, j],
                       ha="center", va="center", color="black", fontsize=16)

plt.colorbar(im, ax=ax3)

# 图4: 各类别的质量分数箱线图
ax4 = axes[1, 1]
data_to_plot = [
    df_compare[df_compare['label'] == 0]['bert_quality'],
    df_compare[df_compare['label'] == 0]['llm_quality'],
    df_compare[df_compare['label'] == 1]['bert_quality'],
    df_compare[df_compare['label'] == 1]['llm_quality'],
]
positions = [1, 2, 4, 5]
bp = ax4.boxplot(data_to_plot, positions=positions, widths=0.6, patch_artist=True)
colors = ['lightblue', 'lightcoral', 'lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax4.set_xticks([1.5, 4.5])
ax4.set_xticklabels(['类别0 (拒识)', '类别1 (寿险)'])
ax4.set_ylabel('质量分数')
ax4.set_title('各类别质量分数分布\n(蓝=BERT, 红=LLM)')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_file = f"{OUTPUT_DIR}/comparison_visualization.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"可视化图表已保存: {plot_file}")

# ============================================
# 第6步：打印高置信度问题样本示例
# ============================================
print("\n" + "=" * 70)
print("高置信度问题样本（前10个）:")
print("=" * 70)

for idx, row in high_confidence_issues.head(10).iterrows():
    label_name = "拒识" if row['label'] == 0 else "寿险相关"
    bert_pred_name = "拒识" if row['bert_pred'] == 0 else "寿险相关"
    llm_pred_name = "拒识" if row['llm_pred'] == 0 else "寿险相关"
    
    print(f"\n样本 #{idx}")
    print(f"  文本: {row['text'][:100]}...")
    print(f"  原始标签: {row['label']} ({label_name})")
    print(f"  BERT预测: {row['bert_pred']} ({bert_pred_name}), 质量分数: {row['bert_quality']:.4f}")
    print(f"  LLM预测: {row['llm_pred']} ({llm_pred_name}), 质量分数: {row['llm_quality']:.4f}")
    print(f"  最低质量分数: {row['min_quality']:.4f}")

print("\n" + "=" * 70)
print("对比分析完成！")
print("=" * 70)
print("\n建议:")
print("1. 优先复核 'high_confidence_issues.csv' 中的样本（两个模型都认为有问题）")
print("2. 查看 'only_bert_issues.csv' 和 'only_llm_issues.csv' 了解模型差异")
print("3. 质量分数相关性高表示两个模型的判断较为一致")
print("4. 可以查看可视化图表了解整体分布情况")








