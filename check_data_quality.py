"""
使用Cleanlab进行数据质量检查 - 检测标注错误和噪声数据

主要功能：
1. 加载标注数据
2. 使用已训练好的BERT模型获取预测概率
3. 使用Cleanlab检测标签错误
4. 生成质量报告和可疑样本列表
"""

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
import warnings
warnings.filterwarnings('ignore')

# ============================================
# 配置参数
# ============================================
DATA_PATH = 'data/intent_data_label.csv'  # 输入数据路径
OUTPUT_DIR = './cleanlab_results'          # 输出目录
MODEL_PATH = "./best_intent_model"         # 已训练好的BERT模型路径
MAX_LENGTH = 128                            # 最大序列长度
BATCH_SIZE = 32                             # 批次大小
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"使用设备: {DEVICE}")
print("=" * 70)

# ============================================
# 第1步：加载和预处理数据
# ============================================
print("第1步：加载数据...")
df = pd.read_csv(DATA_PATH)
print(f"数据总量: {len(df)} 条")

# 数据预处理：合并system和human作为输入文本
def prepare_text(row):
    """将system和human合并为BERT输入"""
    system_col = [col for col in df.columns if 'system' in col.lower()][0]
    human_col = [col for col in df.columns if 'human' in col.lower()][0]
    
    system_text = str(row[system_col]) if pd.notna(row[system_col]) else ""
    human_text = str(row[human_col]) if pd.notna(row[human_col]) else ""
    
    if system_text:
        return f"{system_text} {human_text}"
    else:
        return human_text

df['text'] = df.apply(prepare_text, axis=1)

# 提取标签（二分类：0-拒识, 1-寿险相关）
answer_col = [col for col in df.columns if 'answer' in col.lower() or '答案' in col][0]

def extract_label(answer):
    """
    提取二分类标签：
    0: 拒识（非寿险问题）
    1: 寿险相关
    """
    answer_str = str(answer).lower()
    
    # 判断是否是寿险相关
    if '寿险' in answer_str or 'life insurance' in answer_str or '定期寿' in answer_str or '终身寿' in answer_str:
        return 1
    
    # 其他都是拒识
    return 0

df['label'] = df[answer_col].apply(extract_label)

# 保留有效数据
df_processed = df[['text', 'label']].copy()
df_processed = df_processed[df_processed['text'].str.strip() != '']
df_processed = df_processed.reset_index(drop=True)

print(f"有效数据量: {len(df_processed)} 条")
print(f"\n标签分布:")
print(f"  类别0 (拒识): {(df_processed['label']==0).sum()} 条 ({(df_processed['label']==0).sum()/len(df_processed):.1%})")
print(f"  类别1 (寿险): {(df_processed['label']==1).sum()} 条 ({(df_processed['label']==1).sum()/len(df_processed):.1%})")

# ============================================
# 第2步：加载已训练好的模型
# ============================================
print("\n" + "=" * 70)
print(f"第2步：加载已训练好的BERT模型: {MODEL_PATH}")

# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()  # 设置为评估模式

print("模型加载完成！")

def tokenize_function(examples):
    """BERT分词"""
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH
    )

# ============================================
# 第3步：使用已训练模型获取预测概率
# ============================================
print("\n" + "=" * 70)
print("第3步：使用已训练模型获取所有样本的预测概率...")

# 转换为Dataset格式并分词
dataset = Dataset.from_pandas(df_processed)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 配置推理参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=BATCH_SIZE,
    report_to="none",
)

# 创建Trainer用于批量推理
trainer = Trainer(
    model=model,
    args=training_args,
)

# 获取所有样本的预测
print(f"正在推理 {len(df_processed)} 个样本...")
predictions = trainer.predict(tokenized_dataset)

# 使用softmax获取概率
logits = predictions.predictions
pred_probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
labels = df_processed['label'].values

print("预测完成！")

# ============================================
# 第4步：使用Cleanlab检测标签问题
# ============================================
print("\n" + "=" * 70)
print("第4步：使用Cleanlab分析数据质量...")

# 检测标签错误
label_issues = find_label_issues(
    labels=labels,
    pred_probs=pred_probs,
    return_indices_ranked_by='self_confidence',  # 按置信度排序
    filter_by='both'  # 同时使用多种方法检测
)

# 计算每个样本的标签质量分数（0-1，越低越可能有问题）
quality_scores = get_label_quality_scores(
    labels=labels,
    pred_probs=pred_probs
)

# ============================================
# 第5步：生成质量报告
# ============================================
print("\n" + "=" * 70)
print("第5步：生成数据质量报告...")

# 添加分析结果到DataFrame
df_processed['quality_score'] = quality_scores
df_processed['is_issue'] = False
df_processed.loc[label_issues, 'is_issue'] = True
df_processed['pred_label'] = np.argmax(pred_probs, axis=1)
df_processed['pred_prob_class0'] = pred_probs[:, 0]
df_processed['pred_prob_class1'] = pred_probs[:, 1]

# 统计结果
n_issues = len(label_issues)
issue_rate = n_issues / len(df_processed) * 100

print("\n" + "=" * 70)
print("数据质量检查结果")
print("=" * 70)
print(f"总样本数: {len(df_processed)}")
print(f"检测到问题样本: {n_issues} 条 ({issue_rate:.2f}%)")
print(f"数据质量良好样本: {len(df_processed) - n_issues} 条 ({100-issue_rate:.2f}%)")

# 按类别统计问题
print(f"\n各类别问题统计:")
for label_class in [0, 1]:
    class_name = "拒识" if label_class == 0 else "寿险相关"
    class_total = (df_processed['label'] == label_class).sum()
    class_issues = ((df_processed['label'] == label_class) & (df_processed['is_issue'])).sum()
    class_issue_rate = class_issues / class_total * 100 if class_total > 0 else 0
    print(f"  类别{label_class} ({class_name}): {class_issues}/{class_total} ({class_issue_rate:.2f}%)")

# 质量分数分布
print(f"\n质量分数统计:")
print(f"  平均质量分数: {quality_scores.mean():.4f}")
print(f"  最低质量分数: {quality_scores.min():.4f}")
print(f"  最高质量分数: {quality_scores.max():.4f}")
print(f"  质量分数中位数: {np.median(quality_scores):.4f}")

# ============================================
# 第6步：保存结果
# ============================================
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存完整结果
output_file = f"{OUTPUT_DIR}/data_quality_report.csv"
df_processed.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n完整报告已保存: {output_file}")

# 保存可疑样本（按质量分数从低到高排序）
suspicious_df = df_processed[df_processed['is_issue']].copy()
suspicious_df = suspicious_df.sort_values('quality_score')
suspicious_file = f"{OUTPUT_DIR}/suspicious_samples.csv"
suspicious_df.to_csv(suspicious_file, index=False, encoding='utf-8-sig')
print(f"可疑样本列表已保存: {suspicious_file}")

# 保存质量最差的前100个样本（用于人工复核）
top_issues = df_processed.nsmallest(100, 'quality_score')
top_issues_file = f"{OUTPUT_DIR}/top_100_issues.csv"
top_issues.to_csv(top_issues_file, index=False, encoding='utf-8-sig')
print(f"质量最差的100个样本已保存: {top_issues_file}")

# ============================================
# 第7步：打印典型问题样本
# ============================================
print("\n" + "=" * 70)
print("质量最差的10个样本（建议人工复核）:")
print("=" * 70)

for idx, row in df_processed.nsmallest(10, 'quality_score').iterrows():
    label_name = "拒识" if row['label'] == 0 else "寿险相关"
    pred_name = "拒识" if row['pred_label'] == 0 else "寿险相关"
    
    print(f"\n样本 #{idx}")
    print(f"  文本: {row['text'][:100]}...")
    print(f"  原始标签: {row['label']} ({label_name})")
    print(f"  预测标签: {row['pred_label']} ({pred_name})")
    print(f"  质量分数: {row['quality_score']:.4f}")
    print(f"  预测概率: [拒识: {row['pred_prob_class0']:.3f}, 寿险: {row['pred_prob_class1']:.3f}]")
    print(f"  是否可疑: {'是' if row['is_issue'] else '否'}")

print("\n" + "=" * 70)
print("数据质量检查完成！")
print("=" * 70)
print("\n建议后续操作:")
print("1. 人工复核 'suspicious_samples.csv' 中的可疑样本")
print("2. 重点检查 'top_100_issues.csv' 中质量最差的样本")
print("3. 对于确认错误的样本，进行重新标注或删除")
print("4. 修正后重新训练模型可以显著提升性能")
print("\n文件说明:")
print(f"  - {output_file}: 所有样本的质量分析结果")
print(f"  - {suspicious_file}: Cleanlab检测到的可疑样本")
print(f"  - {top_issues_file}: 质量最差的100个样本（优先复核）")

