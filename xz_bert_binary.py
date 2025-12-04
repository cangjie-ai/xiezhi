# BERT二分类训练脚本（配合Energy-based OOD检测）
# 相比原版xz_bert.py，此版本：
# 1. 只训练两类：寿险相关(1) vs 拒识(0)
# 2. 不需要第三类"other"
# 3. 保存验证集用于后续Energy阈值校准

import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

# ============================================
# 第1步：数据预处理
# ============================================
print("=" * 50)
print("开始加载和预处理数据...")
print("=" * 50)

# 加载CSV数据
df = pd.read_csv('data/intent_data_label.csv')

print("\n原始数据示例：")
print(df.head())
print(f"\n数据集大小: {len(df)} 条")

# 数据预处理：将System和Human组合成BERT的输入
def prepare_text_for_bert(row):
    """将LLM训练数据格式转换为BERT输入格式"""
    system_col = [col for col in df.columns if 'system' in col.lower()][0]
    human_col = [col for col in df.columns if 'human' in col.lower()][0]
    
    system_text = str(row[system_col]) if pd.notna(row[system_col]) else ""
    human_text = str(row[human_col]) if pd.notna(row[human_col]) else ""
    
    if system_text:
        return f"{system_text} {human_text}"
    else:
        return human_text

df['text'] = df.apply(prepare_text_for_bert, axis=1)

# 提取标签：二分类
answer_col = [col for col in df.columns if 'answer' in col.lower() or '答案' in col][0]

def extract_label(answer):
    """
    二分类标签提取：
    0: 拒识（非寿险问题，包括车险、重疾、闲聊等所有非寿险内容）
    1: 寿险相关（关于寿险的专业问题）
    
    ⚠️ 请根据你的实际数据调整判断逻辑！
    """
    answer_str = str(answer).lower()
    
    # 判断是否是寿险相关
    if '寿险' in answer_str or 'life insurance' in answer_str or '定期寿' in answer_str or '终身寿' in answer_str:
        return 1
    
    # 其他都是拒识（包括车险、重疾、闲聊、OOD等）
    return 0

df['label'] = df[answer_col].apply(extract_label)

# 显示处理后的数据
print("\n处理后的数据示例：")
print(df[['text', 'label']].head(10))
print(f"\n标签分布：")
print(df['label'].value_counts())
print(f"  0 (拒识): {(df['label']==0).sum()} 条 ({(df['label']==0).sum()/len(df):.1%})")
print(f"  1 (寿险): {(df['label']==1).sum()} 条 ({(df['label']==1).sum()/len(df):.1%})")

# 只保留需要的列
df_processed = df[['text', 'label']]

# 移除空文本
df_processed = df_processed[df_processed['text'].str.strip() != '']

# ============================================
# 第2步：数据集划分（训练集、验证集、测试集）
# ============================================
# 先划分出测试集（10%）
train_val_df, test_df = train_test_split(
    df_processed, 
    test_size=0.1, 
    random_state=42, 
    stratify=df_processed['label']
)

# 再将训练集划分为训练集和验证集
train_df, val_df = train_test_split(
    train_val_df, 
    test_size=0.15,  # 验证集占总数据的约13.5%
    random_state=42, 
    stratify=train_val_df['label']
)

print(f"\n数据集划分：")
print(f"训练集: {len(train_df)} 条")
print(f"验证集: {len(val_df)} 条")
print(f"测试集: {len(test_df)} 条")

# ⚠️ 重要：保存验证集，用于后续Energy阈值校准
val_df.to_csv('validation_set.csv', index=False)
print(f"\n✓ 验证集已保存至 validation_set.csv（用于Energy阈值校准）")

# 转换为Hugging Face Dataset格式
train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
val_dataset = Dataset.from_pandas(val_df, preserve_index=False)
test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

# ============================================
# 第3步：模型和分词器初始化
# ============================================
# 选择中文BERT模型
checkpoint = "hfl/rbt3"  # 轻量级中文BERT
# 其他选择: "bert-base-chinese", "hfl/chinese-roberta-wwm-ext"

print(f"\n加载预训练模型: {checkpoint}")
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 定义分词函数
def tokenize_function(examples):
    """BERT分词处理"""
    return tokenizer(
        examples["text"], 
        padding="max_length", 
        truncation=True, 
        max_length=128
    )

# 对数据集进行分词处理
print("\n开始分词处理...")
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

print("分词完成！")

# ============================================
# 第4步：加载模型（二分类）
# ============================================
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, 
    num_labels=2,  # 二分类：0-拒识, 1-寿险相关
    problem_type="single_label_classification"
)

# ============================================
# 第5步：定义评估指标
# ============================================
def compute_metrics(eval_pred):
    """计算评估指标：准确率、精确率、召回率、F1分数"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    
    # 计算精确率、召回率、F1（二分类使用binary）
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    # 也计算macro平均（两个类别权重相同）
    _, _, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average='macro', zero_division=0
    )
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_macro": f1_macro,
    }

# ============================================
# 第6步：训练参数配置
# ============================================
training_args = TrainingArguments(
    output_dir="./results_binary",             # 输出目录
    num_train_epochs=50,                       # 最大训练轮次（早停会提前终止）
    per_device_train_batch_size=16,            # 训练批次大小
    per_device_eval_batch_size=16,             # 评估批次大小
    
    # 学习率相关
    learning_rate=2e-5,                        # 学习率
    warmup_ratio=0.1,                          # 预热比例
    
    # 权重衰减（正则化）
    weight_decay=0.01,
    
    # 评估和保存策略
    eval_strategy="epoch",                     # 每个epoch评估一次
    save_strategy="epoch",                     # 每个epoch保存一次
    logging_strategy="steps",                  # 每N步记录一次
    logging_steps=50,                          # 日志记录步数
    
    # 模型保存
    save_total_limit=3,                        # 最多保存3个checkpoint
    load_best_model_at_end=True,               # 训练结束后加载最佳模型
    metric_for_best_model="f1",                # 使用F1分数作为最佳模型标准
    greater_is_better=True,                    # F1越大越好
    
    # 日志目录和TensorBoard
    logging_dir='./logs_binary',
    report_to="tensorboard",
    
    # 设置随机种子
    seed=42,
)

# ============================================
# 第7步：创建Trainer（添加早停回调）
# ============================================
print("\n" + "=" * 50)
print("配置训练器（包含早停机制）")
print("=" * 50)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3,        # 如果3个epoch内F1没提升就停止
            early_stopping_threshold=0.001
        )
    ],
)

# ============================================
# 第8步：开始训练
# ============================================
print("\n" + "=" * 50)
print("开始微调BERT模型（二分类）...")
print("=" * 50)
trainer.train()

# ============================================
# 第9步：保存最优模型
# ============================================
best_model_path = "./best_intent_model"
trainer.save_model(best_model_path)
tokenizer.save_pretrained(best_model_path)
print(f"\n✓ 最佳模型已保存至 {best_model_path}")

# ============================================
# 第10步：在测试集上评估最终性能
# ============================================
print("\n" + "=" * 50)
print("在测试集上评估最终性能...")
print("=" * 50)

test_results = trainer.evaluate(tokenized_test)
print("\n测试集结果：")
for key, value in test_results.items():
    print(f"  {key}: {value:.4f}")

# 获取测试集的预测结果
test_predictions = trainer.predict(tokenized_test)
test_pred_labels = np.argmax(test_predictions.predictions, axis=-1)
test_true_labels = test_predictions.label_ids

# 打印混淆矩阵（2x2）
print("\n混淆矩阵：")
print("标签说明: 0-拒识, 1-寿险相关")
cm = confusion_matrix(test_true_labels, test_pred_labels)
print("           预测0  预测1")
print(f"实际0      {cm[0][0]:5d}  {cm[0][1]:5d}  (拒识)")
print(f"实际1      {cm[1][0]:5d}  {cm[1][1]:5d}  (寿险)")

# 计算每个类别的准确率
print("\n各类别性能：")
for i, label_name in enumerate(['拒识', '寿险相关']):
    if cm[i].sum() > 0:
        class_accuracy = cm[i][i] / cm[i].sum()
        class_recall = cm[i][i] / cm[i].sum()
        class_precision = cm[i][i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        print(f"  {label_name}:")
        print(f"    准确率: {class_accuracy:.4f} (正确{cm[i][i]}/{cm[i].sum()}个)")
        print(f"    召回率: {class_recall:.4f}")
        print(f"    精确率: {class_precision:.4f}")

print("\n" + "=" * 50)
print("训练完成！")
print("=" * 50)

print("\n下一步：")
print("  1. 运行 Energy阈值校准:")
print("     python xz_bert_calibrate_energy.py")
print("\n  2. 或直接运行完整流程:")
print("     python run_energy_calibration.py")
print("\n  3. 查看训练过程可视化:")
print("     tensorboard --logdir=./logs_binary")









