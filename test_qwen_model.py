#!/usr/bin/env python
"""
测试已训练的 Qwen 意图识别模型
用法: python test_qwen_model.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model(model_path, test_questions):
    """测试模型推理"""
    print(f"\n{'='*70}")
    print(f"  加载模型: {model_path}")
    print(f"{'='*70}\n")
    
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"✓ 模型加载完成")
    print(f"✓ EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})\n")
    
    # 测试每个问题
    for i, question in enumerate(test_questions, 1):
        print(f"\n{'-'*70}")
        print(f"测试 {i}/{len(test_questions)}: {question}")
        print(f"{'-'*70}")
        
        # 构建 prompt（与训练时格式完全一致）
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
        
        # 编码
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print(f"Prompt 长度: {inputs['input_ids'].shape[1]} tokens")
        
        # 生成（方案1：贪婪解码）
        print("\n[方案1] 贪婪解码 (do_sample=False):")
        with torch.no_grad():
            outputs_greedy = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens_greedy = outputs_greedy[0][inputs['input_ids'].shape[1]:]
        answer_greedy = tokenizer.decode(new_tokens_greedy, skip_special_tokens=True).strip()
        print(f"  生成: '{answer_greedy}'")
        print(f"  Token IDs: {new_tokens_greedy.tolist()}")
        
        # 生成（方案2：采样）
        print("\n[方案2] 采样 (temperature=0.7):")
        with torch.no_grad():
            outputs_sample = model.generate(
                **inputs,
                max_new_tokens=20,
                min_new_tokens=2,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens_sample = outputs_sample[0][inputs['input_ids'].shape[1]:]
        answer_sample = tokenizer.decode(new_tokens_sample, skip_special_tokens=True).strip()
        print(f"  生成: '{answer_sample}'")
        print(f"  Token IDs: {new_tokens_sample.tolist()}")
        
        # 生成（方案3：更高温度）
        print("\n[方案3] 高温采样 (temperature=1.0):")
        with torch.no_grad():
            outputs_high_temp = model.generate(
                **inputs,
                max_new_tokens=20,
                min_new_tokens=2,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens_high = outputs_high_temp[0][inputs['input_ids'].shape[1]:]
        answer_high = tokenizer.decode(new_tokens_high, skip_special_tokens=True).strip()
        print(f"  生成: '{answer_high}'")
        print(f"  Token IDs: {new_tokens_high.tolist()}")


if __name__ == "__main__":
    # 测试问题列表
    test_questions = [
        "我想买份终身寿险",           # 应该是：寿险相关
        "我今年40岁，买寿险还来得及吗",  # 应该是：寿险相关
        "今天天气不错",              # 应该是：其他
        "请问平安银行几点关门？",      # 应该是：其他
    ]
    
    # 测试合并后的模型
    model_path = "./merged_qwen_intent_model"
    
    try:
        test_model(model_path, test_questions)
    except FileNotFoundError:
        print(f"\n❌ 错误: 找不到模型文件夹 '{model_path}'")
        print("请先运行 xz_qwen0.5b.py 完成训练和合并。")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

