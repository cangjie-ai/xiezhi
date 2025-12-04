"""
1.7B LLM 二分类微调训练脚本
硬件环境：V100 16GB显存（不支持bf16）
数据规模：8000条
目标任务：二分类问题（高精度）
"""

import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import warnings

warnings.filterwarnings("ignore")


# ========================================
# 数据准备模块
# ========================================
def prepare_train_validation_data(test_size=0.15, random_state=42):
    """
    准备训练集和验证集
    
    参数说明：
    - test_size=0.15: 15%作为验证集，85%作为训练集
      原因：对于8000条数据，6800条训练，1200条验证是合理的比例
            验证集足够大以提供可靠的性能评估，同时保留足够的训练数据
    - random_state=42: 随机种子，确保可复现性
    """
    print("=" * 70)
    print("阶段1: 加载并划分数据集")
    print("=" * 70)
    
    # 读取数据
    df = pd.read_csv('data/intent_data_llm.csv', sep='|')
    print(f"✓ 总数据量: {len(df)} 条")
    
    # 打印类别分布
    label_dist = df['answer'].value_counts()
    print(f"✓ 类别分布:\n{label_dist}")
    print(f"✓ 类别比例: {label_dist.values[0]/len(df):.2%} vs {label_dist.values[1]/len(df):.2%}")
    
    # 定义指令模板
    # 这是关键：清晰的任务描述能显著提高模型的分类准确性
    def format_instruction(sample):
        return f"""### 任务
意图分类：判断用户问题是否与寿险业务相关。

### 规则
- 如果问题涉及寿险、终身寿险、定期寿险、寿险保障、寿险理赔等，回答：寿险相关
- 其他所有问题，回答：其他
- 严格只输出这两个答案之一，不要输出其他内容

### 示例
问题：我想买份终身寿险
答案：寿险相关

问题：今天天气怎么样
答案：其他

### 当前任务
问题：{sample['question']}
答案：{sample['answer']}"""
    
    # 应用模板
    df['text'] = df.apply(format_instruction, axis=1)
    
    # 划分训练集和验证集（使用stratify保持类别比例）
    train_df, val_df = train_test_split(
        df, 
        test_size=test_size,
        random_state=random_state,
        stratify=df['answer']  # 保持训练集和验证集的类别比例一致
    )
    
    print(f"\n✓ 训练集大小: {len(train_df)} 条 ({len(train_df)/len(df):.1%})")
    print(f"✓ 验证集大小: {len(val_df)} 条 ({len(val_df)/len(df):.1%})")
    print(f"✓ 训练集类别分布:\n{train_df['answer'].value_counts()}")
    print(f"✓ 验证集类别分布:\n{val_df['answer'].value_counts()}")
    
    # 转换为Hugging Face Dataset格式
    train_dataset = Dataset.from_pandas(train_df[['text', 'question', 'answer']].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[['text', 'question', 'answer']].reset_index(drop=True))
    
    print("\n✓ 训练样本预览:")
    print("-" * 70)
    print(train_df['text'].iloc[0])
    print("-" * 70)
    
    return train_dataset, val_dataset


# ========================================
# 评估指标模块
# ========================================
# 注意：此函数定义为参考，但不能直接用在SFTTrainer中
# 原因：SFTTrainer是为生成式语言建模设计的，输出的是token logits
# 而我们这里是用"生成"的方式做分类（生成"寿险相关"或"其他"）
# 因此需要在训练后单独对验证集进行推理，然后计算分类指标
# 实际的分类指标计算在run_lora_finetune函数的最终评估部分

def compute_metrics_reference(eval_pred):
    """
    计算分类评估指标（参考函数，用于标准分类任务）
    
    指标说明：
    - Accuracy（准确率）：整体预测正确的比例
    - Precision（精确率）：预测为正类中真正为正类的比例
    - Recall（召回率）：真实正类中被正确预测的比例
    - F1-Score：精确率和召回率的调和平均
    
    对于二分类问题，这四个指标能全面评估模型性能
    
    注意：此函数在当前的SFTTrainer设置中不会被调用，
    实际的指标计算在训练后的评估阶段进行（见run_lora_finetune函数）
    """
    predictions, labels = eval_pred
    
    # 对于生成式模型，predictions可能是logits
    # 我们需要提取最可能的token
    if len(predictions.shape) > 1:
        predictions = np.argmax(predictions, axis=-1)
    
    # 计算各项指标
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# ========================================
# 模型配置与训练模块
# ========================================
def run_lora_finetune(train_dataset, val_dataset, model_id="Qwen/Qwen1.5-1.8B-Chat"):
    """
    使用LoRA进行模型微调
    
    参数说明：
    - model_id: 基础模型，这里使用1.7B参数的Qwen模型（接近1.8B）
    """
    print("\n" + "=" * 70)
    print("阶段2: 配置并启动LoRA微调训练")
    print("=" * 70)
    
    # ----------------------------------------
    # 量化配置（BitsAndBytesConfig）
    # ----------------------------------------
    # 使用4-bit量化以最大化显存利用率
    # 原因：1.7B模型在16GB显存上需要激进的内存优化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit量化，相比8-bit节省更多显存
        # 原因：1.7B模型全精度需要约7GB，4-bit量化后约2GB
        
        bnb_4bit_quant_type="nf4",  # 使用NF4（Normal Float 4）量化类型
        # 原因：NF4是专为神经网络权重分布设计的，比标准FP4效果更好
        
        bnb_4bit_use_double_quant=True,  # 双重量化
        # 原因：对量化常数本身再次量化，额外节省0.4GB显存
        
        bnb_4bit_compute_dtype=torch.float16,  # 计算时使用FP16
        # 原因：V100不支持BF16，必须使用FP16；FP16在V100上有硬件加速
    )
    
    # ----------------------------------------
    # 加载分词器
    # ----------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token  # 设置padding token
    tokenizer.padding_side = "right"  # 右侧padding，避免影响生成
    
    print(f"✓ 分词器加载完成")
    print(f"  - EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"  - PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    
    # ----------------------------------------
    # 加载量化模型
    # ----------------------------------------
    print("\n✓ 开始加载量化模型（这可能需要几分钟）...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},  # 全部加载到GPU 0
        trust_remote_code=True,
        torch_dtype=torch.float16,  # V100使用FP16
    )
    print("✓ 模型加载完成")
    
    # 打印模型显存占用
    print(f"✓ 当前GPU显存占用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    # ----------------------------------------
    # LoRA配置（LoraConfig）
    # ----------------------------------------
    # LoRA是Parameter-Efficient Fine-Tuning的一种，只训练少量参数
    lora_config = LoraConfig(
        r=32,  # LoRA秩（rank）
        # 原因：r=32在1.7B模型上能提供足够的表达能力
        # 对于分类任务，较大的r有助于捕捉任务特定的模式
        # 权衡：r越大，可训练参数越多，但仍远小于全参数微调
        
        lora_alpha=64,  # LoRA缩放参数
        # 原因：通常设为2*r，这控制LoRA层的学习强度
        # alpha/r 决定了LoRA更新的缩放比例
        
        target_modules=[
            "q_proj",   # Query投影层
            "k_proj",   # Key投影层
            "v_proj",   # Value投影层
            "o_proj",   # Output投影层
            "gate_proj", # MLP门控层
            "up_proj",   # MLP上投影层
            "down_proj", # MLP下投影层
        ],
        # 原因：覆盖Attention和MLP的所有线性层
        # 对于分类任务，MLP层同样重要，因此全覆盖
        
        lora_dropout=0.1,  # LoRA层的Dropout率
        # 原因：0.1的dropout有助于防止过拟合
        # 8000条数据不算特别多，需要适度的正则化
        
        bias="none",  # 不训练bias
        # 原因：LoRA主要关注权重矩阵，bias参数量少，影响有限
        
        task_type="CAUSAL_LM",  # 因果语言模型任务
    )
    
    # 准备模型进行k-bit训练
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # 打印可训练参数统计
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n✓ LoRA配置完成")
    print(f"  - 可训练参数: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  - 总参数: {total_params:,}")
    
    # ----------------------------------------
    # 训练参数（TrainingArguments）
    # ----------------------------------------
    training_args = TrainingArguments(
        output_dir="./lora_1.7b_classification",  # 输出目录
        
        # --- Batch Size 配置 ---
        per_device_train_batch_size=1,  # 每个GPU的训练batch size
        # 原因：1.7B模型即使量化后，在16GB显存上也需要小batch
        # batch=1是保守选择，确保不会OOM
        
        per_device_eval_batch_size=2,  # 验证时可以稍大
        # 原因：验证时不需要存储梯度，显存占用更小
        
        gradient_accumulation_steps=16,  # 梯度累积步数
        # 原因：有效batch size = 1 * 16 = 16
        # 这是二分类任务的合理batch size
        # 梯度累积不增加显存，但达到大batch的效果
        
        # --- 学习率配置 ---
        learning_rate=2e-4,  # 初始学习率
        # 原因：LoRA微调通常使用较大的学习率（1e-4到5e-4）
        # 2e-4是平衡收敛速度和稳定性的选择
        # 比全参数微调的2e-5大10倍
        
        lr_scheduler_type="cosine",  # 余弦退火学习率调度
        # 原因：cosine调度在训练后期平滑降低学习率
        # 有助于模型收敛到更好的局部最优
        
        warmup_ratio=0.05,  # 预热比例
        # 原因：前5%的训练步骤线性增加学习率
        # 避免初期大学习率导致的不稳定
        
        # --- 训练轮数 ---
        num_train_epochs=5,  # 训练轮数
        # 原因：8000条数据，5个epoch是合理的
        # load_best_model_at_end=True会自动加载最佳模型
        
        # --- 评估与保存策略 ---
        evaluation_strategy="steps",  # 按步数评估
        eval_steps=50,  # 每50步在验证集上评估
        # 原因：6800条训练数据，batch=16，每个epoch约425步
        # 每50步评估一次，每个epoch约评估8次，足够频繁
        
        save_strategy="steps",  # 按步数保存
        save_steps=50,  # 每50步保存一次
        save_total_limit=3,  # 只保留最近3个checkpoint
        # 原因：节省磁盘空间，避免过多checkpoint
        
        load_best_model_at_end=True,  # 训练结束时加载最佳模型
        metric_for_best_model="eval_loss",  # 使用验证损失作为最佳模型指标
        # 原因：对于分类任务，更低的loss通常意味着更好的性能
        
        # --- 精度配置 ---
        fp16=True,  # 使用FP16混合精度训练
        # 原因：V100支持FP16硬件加速，能提速1.5-2倍
        # 同时减少显存占用约50%
        
        bf16=False,  # 不使用BF16
        # 原因：V100不支持BF16，必须设为False
        
        # --- 日志配置 ---
        logging_strategy="steps",
        logging_steps=10,  # 每10步记录一次日志
        # 原因：频繁的日志有助于监控训练进度
        
        report_to="none",  # 不使用外部日志工具（如wandb）
        # 原因：简化配置，使用本地日志即可
        
        # --- 优化器配置 ---
        optim="paged_adamw_32bit",  # 使用分页的AdamW优化器
        # 原因：paged版本能将优化器状态分页到CPU，节省GPU显存
        # 32bit保证优化器状态的精度，避免训练不稳定
        
        weight_decay=0.01,  # 权重衰减（L2正则化）
        # 原因：适度的正则化防止过拟合
        # 0.01是标准值
        
        max_grad_norm=1.0,  # 梯度裁剪阈值
        # 原因：防止梯度爆炸，1.0是常用值
        
        # --- 其他配置 ---
        gradient_checkpointing=True,  # 启用梯度检查点
        # 原因：以计算换显存，将激活值重新计算而非存储
        # 能节省30-40%的显存，训练速度仅降低约20%
        # 对于16GB显存，这是必要的权衡
        
        dataloader_num_workers=4,  # 数据加载器的工作进程数
        # 原因：4个worker能并行预处理数据，避免GPU等待
        
        seed=42,  # 随机种子
        # 原因：确保结果可复现
    )
    
    print(f"\n✓ 训练参数配置完成")
    print(f"  - 有效batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"  - 总训练步数: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps) * training_args.num_train_epochs}")
    print(f"  - 每个epoch步数: {len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
    
    # ----------------------------------------
    # 创建Trainer
    # ----------------------------------------
    def formatting_func(example):
        """格式化函数，提取文本字段"""
        return example["text"]
    
    # 设置模型的pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 注意：SFTTrainer可能与EarlyStoppingCallback有兼容性问题
    # 如果遇到错误，可以移除callbacks参数，直接训练5个epoch
    # load_best_model_at_end=True 会自动加载最佳模型
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,  # 添加验证集
        formatting_func=formatting_func,
        peft_config=lora_config,
        args=training_args,
        max_seq_length=512,  # 最大序列长度
        # 原因：指令模板不长，512足够
        # 较短的序列能减少显存占用
    )
    
    # ----------------------------------------
    # 开始训练
    # ----------------------------------------
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    print(f"预计每个epoch时间: 约10-15分钟（取决于硬件）")
    print(f"预计总训练时间: 约50-75分钟（5个epoch）")
    print(f"训练会自动保存最佳模型（基于验证集loss）")
    print("=" * 70 + "\n")
    
    # 训练前的显存状态
    print(f"训练前GPU显存: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    trainer.train()
    
    print("\n" + "=" * 70)
    print("✓ 训练完成！")
    print("=" * 70)
    
    # ----------------------------------------
    # 保存模型
    # ----------------------------------------
    print("\n✓ 保存LoRA适配器...")
    trainer.save_model("./final_lora_1.7b_adapter")
    tokenizer.save_pretrained("./final_lora_1.7b_adapter")
    print("✓ 模型已保存到: ./final_lora_1.7b_adapter")
    
    # ----------------------------------------
    # 最终评估（使用compute_metrics）
    # ----------------------------------------
    print("\n" + "=" * 70)
    print("阶段3: 在验证集上进行最终评估")
    print("=" * 70)
    
    # 首先获取基础的loss指标
    eval_results = trainer.evaluate()
    print("\n基础评估结果:")
    print("-" * 70)
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    print("-" * 70)
    
    # 然后在验证集上进行推理，计算详细的分类指标
    print("\n计算详细分类指标（这可能需要几分钟）...")
    model.eval()
    
    predictions = []
    true_labels = []
    
    # 定义标签映射
    label_map = {"寿险相关": 1, "其他": 0}
    
    for i in range(len(val_dataset)):
        sample = val_dataset[i]
        question = sample['question']
        true_answer = sample['answer']
        true_labels.append(label_map.get(true_answer, 0))
        
        # 构建推理提示词
        prompt = f"""### 任务
意图分类：判断用户问题是否与寿险业务相关。

### 规则
- 如果问题涉及寿险、终身寿险、定期寿险、寿险保障、寿险理赔等，回答：寿险相关
- 其他所有问题，回答：其他
- 严格只输出这两个答案之一，不要输出其他内容

### 示例
问题：我想买份终身寿险
答案：寿险相关

问题：今天天气怎么样
答案：其他

### 当前任务
问题：{question}
答案："""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案
        if "答案：" in response:
            answer = response.split("答案：")[-1].strip().split('\n')[0].strip()
            if "寿险相关" in answer:
                pred_label = 1
            elif "其他" in answer:
                pred_label = 0
            else:
                pred_label = 0  # 默认为"其他"
        else:
            pred_label = 0
        
        predictions.append(pred_label)
        
        # 每100个样本打印一次进度
        if (i + 1) % 100 == 0:
            print(f"  已处理: {i + 1}/{len(val_dataset)} 样本")
    
    # 使用compute_metrics计算详细指标
    predictions_array = np.array(predictions)
    labels_array = np.array(true_labels)
    
    from sklearn.metrics import classification_report
    
    # 计算各项指标
    accuracy = accuracy_score(labels_array, predictions_array)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array, predictions_array, average='binary', zero_division=0
    )
    
    # 计算混淆矩阵
    cm = confusion_matrix(labels_array, predictions_array)
    
    print("\n" + "=" * 70)
    print("详细分类指标:")
    print("=" * 70)
    print(f"  准确率 (Accuracy):  {accuracy:.4f}")
    print(f"  精确率 (Precision): {precision:.4f}")
    print(f"  召回率 (Recall):    {recall:.4f}")
    print(f"  F1分数 (F1-Score):  {f1:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
    print(f"   [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")
    print(f"\n说明:")
    print(f"  - TN (True Negative):  正确预测为'其他'的样本数")
    print(f"  - FP (False Positive): 错误预测为'寿险相关'的样本数")
    print(f"  - FN (False Negative): 错误预测为'其他'的样本数")
    print(f"  - TP (True Positive):  正确预测为'寿险相关'的样本数")
    print("=" * 70)
    
    # 打印详细的分类报告
    print("\n分类报告:")
    print("-" * 70)
    target_names = ['其他', '寿险相关']
    print(classification_report(labels_array, predictions_array, target_names=target_names))
    print("-" * 70)
    
    # 将分类指标添加到eval_results中
    eval_results['accuracy'] = accuracy
    eval_results['precision'] = precision
    eval_results['recall'] = recall
    eval_results['f1'] = f1
    
    return model, tokenizer, eval_results


# ========================================
# 推理测试模块
# ========================================
def test_model(model, tokenizer, test_questions):
    """
    测试模型推理能力
    
    参数：
    - test_questions: 测试问题列表
    """
    print("\n" + "=" * 70)
    print("阶段4: 模型推理测试")
    print("=" * 70)
    
    model.eval()  # 设置为评估模式
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[测试 {i}/{len(test_questions)}]")
        print(f"问题: {question}")
        
        # 构建推理提示词（与训练格式一致）
        prompt = f"""### 任务
意图分类：判断用户问题是否与寿险业务相关。

### 规则
- 如果问题涉及寿险、终身寿险、定期寿险、寿险保障、寿险理赔等，回答：寿险相关
- 其他所有问题，回答：其他
- 严格只输出这两个答案之一，不要输出其他内容

### 示例
问题：我想买份终身寿险
答案：寿险相关

问题：今天天气怎么样
答案：其他

### 当前任务
问题：{question}
答案："""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # 生成回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,  # 只需要几个token
                temperature=0.1,  # 低温度，更确定性的输出
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码输出
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取答案
        if "答案：" in response:
            answer = response.split("答案：")[-1].strip()
            answer = answer.split('\n')[0].strip()
            if "寿险相关" in answer:
                answer = "寿险相关"
            elif "其他" in answer:
                answer = "其他"
        else:
            answer = response.strip()
        
        print(f"预测: {answer}")
        print("-" * 70)


# ========================================
# 主程序
# ========================================
if __name__ == '__main__':
    # 设置随机种子确保可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 显示GPU信息
    print("=" * 70)
    print("GPU信息")
    print("=" * 70)
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ 总显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"✓ CUDA版本: {torch.version.cuda}")
        print(f"✓ PyTorch版本: {torch.__version__}")
        # 检查BF16支持
        if torch.cuda.is_bf16_supported():
            print("✓ BF16支持: 是（但V100不支持，将使用FP16）")
        else:
            print("✓ BF16支持: 否（将使用FP16）")
    else:
        print("❌ 未检测到GPU，训练将非常慢！")
        exit(1)
    
    # 阶段1: 准备数据
    train_dataset, val_dataset = prepare_train_validation_data()
    
    # 阶段2-3: 训练与评估
    # 注意：这里使用Qwen1.5-1.8B-Chat（最接近1.7B的开源模型）
    # 你也可以替换为其他1.7B模型，如：
    # - "microsoft/phi-2" (2.7B，稍大)
    # - "stabilityai/stablelm-2-1_6b" (1.6B)
    model, tokenizer, eval_results = run_lora_finetune(
        train_dataset, 
        val_dataset,
        model_id="Qwen/Qwen1.5-1.8B-Chat"  # 修改这里以使用不同的基础模型
    )
    
    # 阶段4: 推理测试
    test_questions = [
        "我今年40岁，买寿险还来得及吗？",
        "终身寿险和定期寿险有什么区别？",
        "今天天气怎么样？",
        "附近有什么好吃的餐厅？",
        "寿险理赔需要哪些材料？",
        "如何学习Python编程？"
    ]
    
    test_model(model, tokenizer, test_questions)
    
    print("\n" + "=" * 70)
    print("✓ 所有流程完成！")
    print("=" * 70)
    print(f"\n模型已保存到: ./final_lora_1.7b_adapter")
    print(f"可使用以下代码加载模型进行推理:")
    print("-" * 70)
    print("""
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-1.8B-Chat")
# 加载LoRA适配器
model = PeftModel.from_pretrained(base_model, "./final_lora_1.7b_adapter")
tokenizer = AutoTokenizer.from_pretrained("./final_lora_1.7b_adapter")
    """)
    print("-" * 70)

