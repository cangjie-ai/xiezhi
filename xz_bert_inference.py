# inference.py - 使用训练好的BERT模型进行推理
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

class IntentClassifier:
    """寿险意图分类器"""
    
    def __init__(self, model_path="./best_intent_model"):
        """
        初始化分类器
        Args:
            model_path: 训练好的模型路径
        """
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()  # 设置为评估模式
        
        # 检测是否有GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"使用设备: {self.device}")
        
        # 标签映射（三分类）
        self.label_map = {
            0: "拒识",
            1: "寿险相关",
            2: "other(OOD)"
        }
    
    def predict(self, text, return_prob=False):
        """
        预测单条文本
        Args:
            text: 输入文本（可以是 system + human 拼接）
            return_prob: 是否返回概率分数
        Returns:
            预测标签（和概率）
        """
        # 分词
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # 获取概率
        probs = torch.softmax(logits, dim=-1)
        pred_label = torch.argmax(probs, dim=-1).item()
        pred_prob = probs[0][pred_label].item()
        
        label_name = self.label_map[pred_label]
        
        if return_prob:
            return label_name, pred_prob, {
                "拒识": probs[0][0].item(),
                "寿险相关": probs[0][1].item(),
                "other(OOD)": probs[0][2].item()
            }
        else:
            return label_name
    
    def predict_batch(self, texts, batch_size=32):
        """
        批量预测
        Args:
            texts: 文本列表
            batch_size: 批处理大小
        Returns:
            预测结果列表
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # 分词
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # 获取概率和标签
            probs = torch.softmax(logits, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1).cpu().numpy()
            
            for j, pred_label in enumerate(pred_labels):
                label_name = self.label_map[pred_label]
                confidence = probs[j][pred_label].item()
                results.append({
                    "text": batch_texts[j],
                    "prediction": label_name,
                    "confidence": confidence
                })
        
        return results


def test_single_prediction():
    """测试单条预测"""
    print("=" * 50)
    print("测试单条预测")
    print("=" * 50)
    
    # 初始化分类器
    classifier = IntentClassifier(model_path="./best_intent_model")
    
    # 测试样例（覆盖三类）
    test_cases = [
        "我想了解一下寿险产品",              # 预期：寿险相关
        "你们公司的重疾险怎么样？",          # 预期：拒识或other
        "今天天气真好",                      # 预期：拒识
        "寿险和重疾险有什么区别？",          # 预期：寿险相关或other
        "我要买车险",                        # 预期：拒识
        "定期寿险的保障期限是多久？",        # 预期：寿险相关
        "帮我写一篇作文",                    # 预期：拒识
        "blockchain技术怎么样",              # 预期：other(OOD)
        "终身寿险多少钱",                    # 预期：寿险相关
        "量子计算机的原理",                  # 预期：other(OOD)
    ]
    
    print("\n预测结果：")
    print("-" * 80)
    for text in test_cases:
        label, prob, all_probs = classifier.predict(text, return_prob=True)
        print(f"文本: {text}")
        print(f"预测: {label} (置信度: {prob:.4f})")
        print(f"详细概率: 拒识={all_probs['拒识']:.4f}, 寿险={all_probs['寿险相关']:.4f}, other={all_probs['other(OOD)']:.4f}")
        print("-" * 80)


def test_batch_prediction():
    """测试批量预测"""
    print("\n" + "=" * 50)
    print("测试批量预测")
    print("=" * 50)
    
    # 初始化分类器
    classifier = IntentClassifier(model_path="./best_intent_model")
    
    # 从CSV加载测试数据（可选）
    # df = pd.read_csv('test_data.csv')
    # texts = df['text'].tolist()
    
    texts = [
        "我想购买定期寿险",
        "今天股票大涨",
        "寿险的保费怎么算？",
        "你好",
        "终身寿险和定期寿险哪个好？",
    ]
    
    # 批量预测
    results = classifier.predict_batch(texts, batch_size=32)
    
    # 显示结果
    print("\n批量预测结果：")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text'][:30]}")
        print(f"   预测: {result['prediction']} (置信度: {result['confidence']:.4f})")
    
    # 可选：保存结果到CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('prediction_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ 预测结果已保存至 prediction_results.csv")


def test_with_system_prompt():
    """测试带System Prompt的预测"""
    print("\n" + "=" * 50)
    print("测试带System Prompt的预测")
    print("=" * 50)
    
    classifier = IntentClassifier(model_path="./best_intent_model")
    
    # 模拟System + Human格式
    system_prompt = "你是一个保险领域的智能客服助手"
    human_questions = [
        "我想了解寿险",
        "今天天气怎么样",
        "什么是定期寿险？"
    ]
    
    print("\n预测结果：")
    print("-" * 50)
    for question in human_questions:
        # 拼接System和Human（与训练时一致）
        full_text = f"{system_prompt} {question}"
        label, prob, _ = classifier.predict(full_text, return_prob=True)
        print(f"问题: {question}")
        print(f"预测: {label} (置信度: {prob:.4f})")
        print("-" * 50)


if __name__ == "__main__":
    # 运行测试
    test_single_prediction()
    test_batch_prediction()
    test_with_system_prompt()
    
    print("\n" + "=" * 50)
    print("测试完成！")
    print("=" * 50)

