# Energy-based OOD检测推理器
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Tuple, Dict, List, Optional

class EnergyBasedIntentClassifier:
    """
    基于Energy的OOD检测分类器
    核心思想：用Energy = -log(sum(exp(logits)))来判断样本是否OOD
    - ID样本（训练集内）：Energy低
    - OOD样本（训练集外）：Energy高
    """
    
    def __init__(self, 
                 model_path: str = "./best_intent_model",
                 energy_threshold_high: float = 1.5,
                 energy_threshold_low: float = 0.3,
                 temperature: float = 1.0):
        """
        初始化分类器
        Args:
            model_path: 训练好的BERT模型路径
            energy_threshold_high: Energy超过此值判定为OOD（需要在验证集上校准）
            energy_threshold_low: Energy低于此值判定为确定样本（需要在验证集上校准）
            temperature: 温度参数，通常设为1.0
        """
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # 检测设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"使用设备: {self.device}")
        
        # Energy阈值参数
        self.energy_threshold_high = energy_threshold_high
        self.energy_threshold_low = energy_threshold_low
        self.temperature = temperature
        
        # 标签映射（假设二分类）
        self.label_map = {
            0: "拒识",
            1: "寿险相关"
        }
        
        print(f"Energy阈值配置: low={energy_threshold_low:.3f}, high={energy_threshold_high:.3f}")
    
    def compute_energy(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        计算Energy分数
        公式: Energy = -T * log(sum(exp(logits/T)))
             = -T * logsumexp(logits/T)
        
        Args:
            logits: [batch_size, num_classes] 模型输出的logits
            temperature: 温度参数
        Returns:
            energy: [batch_size] 每个样本的energy值
        """
        energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
        return energy
    
    def predict_single(self, 
                      text: str, 
                      return_details: bool = False) -> Tuple:
        """
        预测单条文本（带Energy检测）
        
        Args:
            text: 输入文本
            return_details: 是否返回详细信息（energy、logits等）
        
        Returns:
            如果return_details=False: (label, confidence_level)
            如果return_details=True: (label, confidence_level, details_dict)
        """
        # 分词
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 前向传播获取logits
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, num_classes]
        
        # 计算Energy
        energy = self.compute_energy(logits, self.temperature).item()
        
        # 计算softmax概率（仅供参考，不用于决策）
        probs = F.softmax(logits, dim=-1)
        pred_label_idx = torch.argmax(probs, dim=-1).item()
        pred_prob = probs[0][pred_label_idx].item()
        
        # Energy阈值判断
        if energy > self.energy_threshold_high:
            # Energy过高 → OOD样本
            final_label = "拒识（OOD）"
            confidence = "low"
            decision_reason = f"Energy={energy:.3f} > {self.energy_threshold_high:.3f}, 判定为OOD"
            
        elif energy < self.energy_threshold_low:
            # Energy很低 → 确定的ID样本
            final_label = self.label_map[pred_label_idx]
            confidence = "high"
            decision_reason = f"Energy={energy:.3f} < {self.energy_threshold_low:.3f}, 模型确定"
            
        else:
            # Energy在中间区域 → 不确定，使用模型预测但标记中等置信度
            final_label = self.label_map[pred_label_idx]
            confidence = "medium"
            decision_reason = f"Energy={energy:.3f} 在阈值区间，不确定（建议LLM验证）"
        
        if return_details:
            details = {
                "energy": energy,
                "logits": logits[0].cpu().tolist(),
                "probs": probs[0].cpu().tolist(),
                "pred_label_idx": pred_label_idx,
                "reason": decision_reason,
            }
            return final_label, confidence, details
        else:
            return final_label, confidence
    
    def predict_batch(self, 
                     texts: List[str], 
                     batch_size: int = 32) -> List[Dict]:
        """
        批量预测（带Energy检测）
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
        
        Returns:
            结果列表，每个元素为字典: {
                "text": 输入文本,
                "prediction": 预测标签,
                "confidence": 置信度等级,
                "energy": Energy值,
                "reason": 决策原因
            }
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
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 预测
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # 计算Energy
            energies = self.compute_energy(logits, self.temperature).cpu().numpy()
            
            # 计算概率和标签
            probs = F.softmax(logits, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1).cpu().numpy()
            pred_probs = probs[torch.arange(len(pred_labels)), pred_labels].cpu().numpy()
            
            # 逐个样本判断
            for j in range(len(batch_texts)):
                energy = energies[j]
                pred_label_idx = pred_labels[j]
                
                # Energy阈值判断
                if energy > self.energy_threshold_high:
                    final_label = "拒识（OOD）"
                    confidence = "low"
                    reason = f"Energy={energy:.3f} > threshold_high"
                elif energy < self.energy_threshold_low:
                    final_label = self.label_map[pred_label_idx]
                    confidence = "high"
                    reason = f"Energy={energy:.3f} < threshold_low"
                else:
                    final_label = self.label_map[pred_label_idx]
                    confidence = "medium"
                    reason = f"Energy={energy:.3f} in uncertain zone"
                
                results.append({
                    "text": batch_texts[j],
                    "prediction": final_label,
                    "confidence": confidence,
                    "energy": float(energy),
                    "reason": reason
                })
        
        return results
    
    def update_thresholds(self, threshold_low: float, threshold_high: float):
        """
        更新Energy阈值（用于动态调整）
        
        Args:
            threshold_low: 新的低阈值
            threshold_high: 新的高阈值
        """
        self.energy_threshold_low = threshold_low
        self.energy_threshold_high = threshold_high
        print(f"阈值已更新: low={threshold_low:.3f}, high={threshold_high:.3f}")


def test_single_prediction():
    """测试单条预测"""
    print("=" * 80)
    print("测试Energy-based OOD检测")
    print("=" * 80)
    
    # 初始化分类器（使用临时阈值，实际阈值需要用校准脚本计算）
    classifier = EnergyBasedIntentClassifier(
        model_path="./best_intent_model",
        energy_threshold_high=1.5,  # 临时值，需要校准
        energy_threshold_low=0.3    # 临时值，需要校准
    )
    
    # 测试样例（覆盖ID和OOD）
    test_cases = [
        # 预期ID样本（寿险相关）
        "我想了解一下定期寿险产品",
        "寿险保费怎么计算的？",
        "终身寿险和定期寿险有什么区别？",
        "寿险能保障到多大年龄？",
        
        # 预期ID样本（拒识）
        "你们公司的重疾险怎么样？",  # 非寿险保险
        "我要买车险",
        
        # 预期OOD样本（明显无关）
        "今天星期几",
        "区块链技术介绍",
        "帮我写一篇作文",
        "量子计算机的工作原理",
        "今天天气真好",
        "推荐一部好看的电影",
    ]
    
    print("\n预测结果：")
    print("-" * 80)
    
    for text in test_cases:
        label, confidence, details = classifier.predict_single(text, return_details=True)
        
        print(f"输入: {text}")
        print(f"预测: {label} | 置信度: {confidence}")
        print(f"Energy: {details['energy']:.4f}")
        print(f"Logits: [{', '.join([f'{x:.3f}' for x in details['logits']])}]")
        print(f"Softmax概率: [{', '.join([f'{x:.3f}' for x in details['probs']])}]")
        print(f"决策原因: {details['reason']}")
        print("-" * 80)


def test_batch_prediction():
    """测试批量预测"""
    print("\n" + "=" * 80)
    print("测试批量预测")
    print("=" * 80)
    
    classifier = EnergyBasedIntentClassifier(
        model_path="./best_intent_model",
        energy_threshold_high=1.5,
        energy_threshold_low=0.3
    )
    
    texts = [
        "我想购买定期寿险",
        "今天股票大涨",
        "寿险的保费怎么算？",
        "你好",
        "终身寿险和定期寿险哪个好？",
        "帮我订个外卖",
        "寿险能保障哪些情况？",
        "区块链是什么",
    ]
    
    results = classifier.predict_batch(texts, batch_size=4)
    
    print("\n批量预测结果：")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['text']}")
        print(f"   预测: {result['prediction']} (置信度: {result['confidence']})")
        print(f"   Energy: {result['energy']:.4f}")
        print(f"   原因: {result['reason']}")


if __name__ == "__main__":
    # 运行测试
    test_single_prediction()
    test_batch_prediction()
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    print("\n提示：当前使用的是临时阈值，请运行 xz_bert_calibrate_energy.py 来计算最佳阈值")









