import torch
import pandas as pd
from datasets import Dataset
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

# 心法: 数据是龙的“草料”，格式必须符合它的胃口。
# 招式: 构建对话式指令数据集
def prepare_instruction_data():
    print("--- 阶段1: 构建指令微调数据集 ---")
    # 对于LLM，我们不再是简单的 (text, label) 对，而是通过指令和对话来教它。
    df = pd.read_csv('data/intent_data_llm.csv', sep='|')  # 指定分隔符为竖线

    # 我们需要将数据转换成模型喜欢的格式，通常是包含特定系统提示的文本。
    # 这是最关键的一步，称为"指令模板化" (Prompt Templating)。
    def format_instruction(sample):
        # 使用更明确的指令和 few-shot 示例
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

    df['text'] = df.apply(format_instruction, axis=1)
    print("指令数据预览:\n", df['text'].iloc[0])
    
    # 转换为Hugging Face的Dataset对象
    return Dataset.from_pandas(df)

# 心法: QLoRA是核心内功，配置好参数，才能“四两拨千斤”。
def run_qlora_finetune(dataset):
    print("\n--- 阶段2: 执行QLoRA微调 ---")
    
    # 模型ID
    model_id = "Qwen/Qwen1.5-0.5B-Chat"
    
    # 招式: 4位量化配置 (BitsAndBytesConfig)
    # 这是开启QLoRA的第一步：告诉模型以4位加载。
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4", # 使用NF4类型进行量化，效果更好
        bnb_4bit_compute_dtype=torch.bfloat16 # 在计算时，使用bfloat16以保持精度和速度
    )
    
    # 招式: 加载量化后的模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # ChatML模板要求pad_token设为eos_token
    tokenizer.pad_token = tokenizer.eos_token
    
    # 打印 tokenizer 信息
    print(f"分词器加载完成. EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    print(f"PAD token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id})")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"":0} # 将模型自动分配到当前可用GPU
    )
    
    # 招式: 配置LoRA参数 (LoraConfig)
    # 这里定义了我们的“外骨骼”有多大，以及要附加在模型的哪些部位。
    lora_config = LoraConfig(
        r=16,  # LoRA的秩，可以理解为“外骨骼”的复杂度，通常设为8, 16, 32
        lora_alpha=32, # LoRA的缩放因子
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # 指定要附加LoRA的模块，通常是注意力层
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 准备模型进行k-bit训练并应用PEFT配置
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    
    # 招式: 配置训练参数 (TrainingArguments)
    training_args = TrainingArguments(
        output_dir="./qwen_intent_finetuned",
        per_device_train_batch_size=2, # Batch size减小，适应显存
        gradient_accumulation_steps=4, # 梯度累积，等效于 batch_size = 2 * 4 = 8
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="epoch",
        bf16=True, # 如果你的显卡支持，使用bfloat16能加速并节省显存
    )
    
    # 招式: 使用SFTTrainer进行监督微调
    # 定义格式化函数（新版本API）
    def formatting_func(example):
        return example["text"]
    
    # 新版本 SFTTrainer API 说明:
    # - tokenizer 参数已移除，从 model 自动获取或需要手动设置到 model.config
    # - peft_config 仍然支持
    # - formatting_func 替代了 dataset_text_field
    
    # 将 tokenizer 附加到 model（新版本需要）
    model.config.pad_token_id = tokenizer.pad_token_id
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,  # 文本格式化函数
        peft_config=lora_config,          # LoRA 配置（保留）
        args=training_args,                # 训练参数
    )
    
    print("开始训练...")
    trainer.train()
    print("训练完成！")
    
    # 保存最终的LoRA适配器
    trainer.save_model("./final_lora_adapter")
    return model, tokenizer

# 心法: “人剑合一”，将学到的内力（Adapter）融入剑身（Base Model）。
def merge_and_test(base_model_id, adapter_dir):
    print("\n--- 阶段3: 合并LoRA适配器并测试 ---")
    
    # 加载原始的、未经量化的基础模型
    # 注意：合并需要足够的CPU内存
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✓ 加载分词器. EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")
    
    # 从PEFT模型加载并合并LoRA权重
    # 这会将adapter_dir中的LoRA权重与base_model的权重合并
    from peft import PeftModel
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = merged_model.merge_and_unload() # 关键步骤：合并权重并卸载LoRA层
    
    print("模型合并完成！现在它是一个独立的、微调过的模型。")
    
    # 保存合并后的完整模型，以便将来直接加载使用
    merged_model.save_pretrained("./merged_qwen_intent_model")
    tokenizer.save_pretrained("./merged_qwen_intent_model")
    
    # --- 测试 ---
    test_question = "我今年40岁，买寿险还来得及吗"
    # test_question = "附近有什么好吃的？"
    
    # 构建与训练时一致的推理模板（这很重要！）
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
问题：{test_question}
答案："""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(merged_model.device)
    
    print("开始推理...")
    print(f"Prompt 长度: {inputs['input_ids'].shape[1]} tokens")
    
    # 生成配置优化
    outputs = merged_model.generate(
        **inputs,
        max_new_tokens=50,              # 增加生成长度，给模型更多空间
        min_new_tokens=2,               # 至少生成2个token
        do_sample=True,                 # 启用采样（更多样化的输出）
        temperature=0.7,                # 温度：控制随机性（0.7较保守）
        top_p=0.9,                      # nucleus sampling
        top_k=50,                       # top-k sampling
        repetition_penalty=1.1,         # 惩罚重复
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print(f"生成长度: {outputs.shape[1]} tokens (新生成: {outputs.shape[1] - inputs['input_ids'].shape[1]})")
    
    # 调试：打印新生成的 token IDs
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    print(f"\n[调试] 新生成的 token IDs: {new_tokens.tolist()}")
    print(f"[调试] 解码新 tokens: '{tokenizer.decode(new_tokens, skip_special_tokens=False)}'")
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 从完整输出中提取答案
    if "答案：" in response:
        # 提取最后一个 "答案：" 后面的内容
        answer = response.split("答案：")[-1].strip()
        # 只取第一行，去除换行后的多余内容
        answer = answer.split('\n')[0].strip()
        # 进一步清理，只保留 "寿险相关" 或 "其他"
        if "寿险相关" in answer:
            answer = "寿险相关"
        elif "其他" in answer:
            answer = "其他"
    else:
        # 如果格式不符合预期，尝试直接匹配关键词
        if "寿险相关" in response:
            answer = "寿险相关"
        elif "其他" in response:
            answer = "其他"
        else:
            answer = response.strip()
            print("⚠️ 警告: 模型输出格式不符合预期，返回完整输出")
    
    print(f"\n{'='*70}")
    print(f"输入问题: '{test_question}'")
    print(f"预测意图: '{answer}'")
    print(f"{'='*70}")
    print(f"\n完整响应:\n{'-'*70}\n{response}\n{'-'*70}")


if __name__ == '__main__':
    # 套路一
    instruction_dataset = prepare_instruction_data()
    
    # 套路二
    trained_model, tokenizer = run_qlora_finetune(instruction_dataset)
    
    # 清理显存，为合并模型做准备
    del trained_model
    torch.cuda.empty_cache()

    # 套路二的收尾式
    merge_and_test(base_model_id="Qwen/Qwen1.5-0.5B-Chat", adapter_dir="./final_lora_adapter")