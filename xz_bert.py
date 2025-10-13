# fine_tune.py
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score

# 加载CSV数据
raw_datasets = load_dataset('csv', data_files='data/intent_data_label.csv')

# 模型选型
checkpoint = "hfl/rbt3"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# 定义分词函数
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

# 对整个数据集进行分词处理
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# 移除不需要的列，重命名label列
tokenized_datasets = tokenized_datasets['train'].train_test_split(test_size=0.2)
print("数据准备完毕:", tokenized_datasets)


# 加载预训练模型，并告知它我们要做的是一个2分类任务
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

# 定义训练参数
training_args = TrainingArguments(
    output_dir="./results",          # 输出目录
    num_train_epochs=10,              # 训练轮次
    per_device_train_batch_size=8,   # 训练批次大小
    per_device_eval_batch_size=8,    # 评估批次大小
    warmup_steps=10,                 # 预热步数
    weight_decay=0.01,               # 权重衰减
    logging_dir='./logs',            # 日志目录
    eval_strategy="epoch",           # 每个epoch评估一次
    save_strategy="epoch",           # 每个epoch保存一次
    load_best_model_at_end=True,     # 训练结束后加载最佳模型
)

# 创建Trainer实例
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# 开始训练！
print("开始微调模型...")
trainer.train()

# 保存最好的模型
best_model_path = "./best_intent_model"
trainer.save_model(best_model_path)
tokenizer.save_pretrained(best_model_path)
print(f"最佳模型已保存至 {best_model_path}")