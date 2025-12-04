# Energy阈值校准工具
# 用于在验证集上计算最佳Energy阈值
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
import json
from pathlib import Path

class EnergyThresholdCalibrator:
    """Energy阈值校准器 - 在验证集上计算最佳阈值"""
    
    def __init__(self, model_path: str = "./best_intent_model", temperature: float = 1.0):
        """
        初始化校准器
        Args:
            model_path: 训练好的模型路径
            temperature: 温度参数
        """
        print(f"加载模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"使用设备: {self.device}")
        
        self.temperature = temperature
    
    def compute_energy(self, logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        计算Energy分数
        Energy = -T * log(sum(exp(logits/T)))
        """
        energy = -temperature * torch.logsumexp(logits / temperature, dim=-1)
        return energy
    
    def compute_energies_from_texts(self, 
                                   texts: List[str], 
                                   batch_size: int = 32,
                                   show_progress: bool = True) -> np.ndarray:
        """
        批量计算文本的Energy值
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
        
        Returns:
            energies: numpy array [len(texts)]
        """
        energies = []
        
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="计算Energy")
        
        for i in iterator:
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
            
            # 前向传播
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # 计算Energy
            batch_energies = self.compute_energy(logits, self.temperature).cpu().numpy()
            energies.extend(batch_energies)
        
        return np.array(energies)
    
    def calibrate_from_dataframe(self, 
                                df: pd.DataFrame,
                                text_col: str = 'text',
                                label_col: str = 'label',
                                batch_size: int = 32) -> Dict:
        """
        从DataFrame校准阈值（适用于有标签的验证集）
        
        Args:
            df: 包含文本和标签的DataFrame
            text_col: 文本列名
            label_col: 标签列名（0=拒识, 1=寿险相关, 或者其他二分类）
            batch_size: 批处理大小
        
        Returns:
            校准结果字典
        """
        print(f"\n开始校准Energy阈值...")
        print(f"数据集大小: {len(df)} 条")
        print(f"标签分布:\n{df[label_col].value_counts()}")
        
        # 计算所有样本的Energy
        texts = df[text_col].tolist()
        labels = df[label_col].values
        
        energies = self.compute_energies_from_texts(texts, batch_size=batch_size)
        
        # 分析Energy分布
        print("\n" + "=" * 80)
        print("Energy统计分析")
        print("=" * 80)
        
        # 整体统计
        print(f"\n整体数据 (n={len(energies)}):")
        print(f"  均值: {energies.mean():.4f}")
        print(f"  标准差: {energies.std():.4f}")
        print(f"  最小值: {energies.min():.4f}")
        print(f"  最大值: {energies.max():.4f}")
        print(f"  中位数: {np.median(energies):.4f}")
        
        # 分标签统计
        unique_labels = np.unique(labels)
        label_stats = {}
        
        for label in unique_labels:
            mask = labels == label
            label_energies = energies[mask]
            
            stats = {
                "count": len(label_energies),
                "mean": float(label_energies.mean()),
                "std": float(label_energies.std()),
                "min": float(label_energies.min()),
                "max": float(label_energies.max()),
                "median": float(np.median(label_energies)),
                "percentile_5": float(np.percentile(label_energies, 5)),
                "percentile_25": float(np.percentile(label_energies, 25)),
                "percentile_75": float(np.percentile(label_energies, 75)),
                "percentile_95": float(np.percentile(label_energies, 95)),
                "percentile_99": float(np.percentile(label_energies, 99)),
            }
            label_stats[int(label)] = stats
            
            print(f"\n标签={label} (n={stats['count']}):")
            print(f"  均值±标准差: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"  范围: [{stats['min']:.4f}, {stats['max']:.4f}]")
            print(f"  分位数: [5%={stats['percentile_5']:.4f}, 25%={stats['percentile_25']:.4f}, "
                  f"75%={stats['percentile_75']:.4f}, 95%={stats['percentile_95']:.4f}]")
        
        # 推荐阈值策略
        print("\n" + "=" * 80)
        print("推荐阈值策略")
        print("=" * 80)
        
        # 策略1：基于所有ID样本的分位数（最常用）
        all_id_energies = energies  # 假设所有验证集样本都是ID（训练集内）
        
        threshold_low_conservative = np.percentile(all_id_energies, 10)  # 保守：10%分位
        threshold_low_moderate = np.percentile(all_id_energies, 25)     # 适中：25%分位
        threshold_low_aggressive = np.percentile(all_id_energies, 50)    # 激进：50%分位（中位数）
        
        threshold_high_aggressive = np.percentile(all_id_energies, 90)   # 激进：90%分位
        threshold_high_moderate = np.percentile(all_id_energies, 95)     # 适中：95%分位
        threshold_high_conservative = np.percentile(all_id_energies, 99) # 保守：99%分位
        
        recommendations = {
            "conservative": {
                "threshold_low": float(threshold_low_conservative),
                "threshold_high": float(threshold_high_conservative),
                "description": "保守策略 - 尽量避免误拒ID样本，但可能放过一些OOD"
            },
            "moderate": {
                "threshold_low": float(threshold_low_moderate),
                "threshold_high": float(threshold_high_moderate),
                "description": "适中策略 - 平衡误拒和误接受（推荐）"
            },
            "aggressive": {
                "threshold_low": float(threshold_low_aggressive),
                "threshold_high": float(threshold_high_aggressive),
                "description": "激进策略 - 严格拒绝可疑样本，但可能误拒一些ID样本"
            }
        }
        
        print("\n策略1：基于分位数（推荐用于没有真实OOD样本的情况）")
        print("-" * 80)
        for strategy, params in recommendations.items():
            print(f"\n{strategy.upper()} - {params['description']}")
            print(f"  threshold_low  = {params['threshold_low']:.4f}")
            print(f"  threshold_high = {params['threshold_high']:.4f}")
        
        # 策略2：基于标准差（可选）
        mean_energy = all_id_energies.mean()
        std_energy = all_id_energies.std()
        
        threshold_low_1std = mean_energy - 1.0 * std_energy
        threshold_high_2std = mean_energy + 2.0 * std_energy
        threshold_high_3std = mean_energy + 3.0 * std_energy
        
        print("\n\n策略2：基于标准差（适用于Energy近似正态分布的情况）")
        print("-" * 80)
        print(f"均值 = {mean_energy:.4f}, 标准差 = {std_energy:.4f}")
        print(f"\n1-sigma策略:")
        print(f"  threshold_low  = {threshold_low_1std:.4f} (均值 - 1σ)")
        print(f"  threshold_high = {threshold_high_2std:.4f} (均值 + 2σ)")
        print(f"\n2-sigma策略:")
        print(f"  threshold_low  = {threshold_low_1std:.4f} (均值 - 1σ)")
        print(f"  threshold_high = {threshold_high_3std:.4f} (均值 + 3σ)")
        
        # 汇总结果
        calibration_result = {
            "model_path": str(self.model_path) if hasattr(self, 'model_path') else "unknown",
            "dataset_size": int(len(df)),
            "temperature": float(self.temperature),
            "overall_stats": {
                "mean": float(energies.mean()),
                "std": float(energies.std()),
                "min": float(energies.min()),
                "max": float(energies.max()),
                "median": float(np.median(energies))
            },
            "label_stats": label_stats,
            "recommendations": recommendations,
            "std_based": {
                "mean": float(mean_energy),
                "std": float(std_energy),
                "threshold_low_1std": float(threshold_low_1std),
                "threshold_high_2std": float(threshold_high_2std),
                "threshold_high_3std": float(threshold_high_3std)
            }
        }
        
        return calibration_result
    
    def calibrate_with_ood_samples(self,
                                  id_texts: List[str],
                                  ood_texts: List[str],
                                  batch_size: int = 32) -> Dict:
        """
        使用真实OOD样本校准阈值（如果有的话）
        
        Args:
            id_texts: ID样本（训练集内）文本列表
            ood_texts: OOD样本文本列表
            batch_size: 批处理大小
        
        Returns:
            校准结果字典
        """
        print(f"\n开始校准Energy阈值（带真实OOD样本）...")
        print(f"ID样本数: {len(id_texts)}")
        print(f"OOD样本数: {len(ood_texts)}")
        
        # 计算ID和OOD样本的Energy
        print("\n计算ID样本Energy...")
        id_energies = self.compute_energies_from_texts(id_texts, batch_size=batch_size)
        
        print("计算OOD样本Energy...")
        ood_energies = self.compute_energies_from_texts(ood_texts, batch_size=batch_size)
        
        # 统计分析
        print("\n" + "=" * 80)
        print("Energy分布对比")
        print("=" * 80)
        
        print(f"\nID样本 (n={len(id_energies)}):")
        print(f"  均值±标准差: {id_energies.mean():.4f} ± {id_energies.std():.4f}")
        print(f"  范围: [{id_energies.min():.4f}, {id_energies.max():.4f}]")
        print(f"  分位数: [25%={np.percentile(id_energies, 25):.4f}, "
              f"50%={np.percentile(id_energies, 50):.4f}, "
              f"75%={np.percentile(id_energies, 75):.4f}, "
              f"95%={np.percentile(id_energies, 95):.4f}]")
        
        print(f"\nOOD样本 (n={len(ood_energies)}):")
        print(f"  均值±标准差: {ood_energies.mean():.4f} ± {ood_energies.std():.4f}")
        print(f"  范围: [{ood_energies.min():.4f}, {ood_energies.max():.4f}]")
        print(f"  分位数: [5%={np.percentile(ood_energies, 5):.4f}, "
              f"25%={np.percentile(ood_energies, 25):.4f}, "
              f"50%={np.percentile(ood_energies, 50):.4f}, "
              f"75%={np.percentile(ood_energies, 75):.4f}]")
        
        # 计算分离度
        separation = ood_energies.mean() - id_energies.mean()
        print(f"\n分离度 (OOD均值 - ID均值): {separation:.4f}")
        if separation > 0:
            print("✓ OOD样本的Energy显著高于ID样本（符合预期）")
        else:
            print("⚠ 警告：OOD样本的Energy未高于ID样本，模型可能无法有效区分OOD")
        
        # 寻找最佳阈值（最大化TPR和TNR之和）
        print("\n" + "=" * 80)
        print("寻找最佳分割阈值")
        print("=" * 80)
        
        # 在ID和OOD的范围内搜索最佳阈值
        all_energies = np.concatenate([id_energies, ood_energies])
        candidate_thresholds = np.linspace(all_energies.min(), all_energies.max(), 100)
        
        best_threshold = None
        best_score = -1
        best_tpr = 0
        best_tnr = 0
        
        for threshold in candidate_thresholds:
            # TPR: ID样本被正确接受的比例（Energy < threshold）
            tpr = (id_energies < threshold).mean()
            # TNR: OOD样本被正确拒绝的比例（Energy >= threshold）
            tnr = (ood_energies >= threshold).mean()
            # 综合得分
            score = tpr + tnr
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_tpr = tpr
                best_tnr = tnr
        
        print(f"\n最佳阈值（最大化TPR+TNR）:")
        print(f"  threshold_high = {best_threshold:.4f}")
        print(f"  ID接受率(TPR) = {best_tpr:.2%}")
        print(f"  OOD拒绝率(TNR) = {best_tnr:.2%}")
        print(f"  综合得分 = {best_score:.4f}")
        
        # 其他推荐阈值
        threshold_high_95 = np.percentile(id_energies, 95)  # 拒绝5%的ID样本
        threshold_high_99 = np.percentile(id_energies, 99)  # 拒绝1%的ID样本
        
        print(f"\n其他推荐:")
        print(f"  95%分位阈值 = {threshold_high_95:.4f} (保留95%的ID样本)")
        print(f"  99%分位阈值 = {threshold_high_99:.4f} (保留99%的ID样本)")
        
        result = {
            "id_stats": {
                "count": int(len(id_energies)),
                "mean": float(id_energies.mean()),
                "std": float(id_energies.std()),
                "min": float(id_energies.min()),
                "max": float(id_energies.max())
            },
            "ood_stats": {
                "count": int(len(ood_energies)),
                "mean": float(ood_energies.mean()),
                "std": float(ood_energies.std()),
                "min": float(ood_energies.min()),
                "max": float(ood_energies.max())
            },
            "separation": float(separation),
            "best_threshold": {
                "value": float(best_threshold),
                "tpr": float(best_tpr),
                "tnr": float(best_tnr),
                "score": float(best_score)
            },
            "alternative_thresholds": {
                "percentile_95": float(threshold_high_95),
                "percentile_99": float(threshold_high_99)
            }
        }
        
        return result
    
    def save_calibration_result(self, result: Dict, output_path: str = "energy_thresholds.json"):
        """
        保存校准结果到JSON文件
        
        Args:
            result: 校准结果字典
            output_path: 输出文件路径
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n✓ 校准结果已保存至: {output_path}")


def main_calibrate_from_validation_set():
    """
    主函数：从验证集CSV文件校准阈值
    适用场景：有训练集划分出的验证集，但没有真实OOD样本
    """
    print("=" * 80)
    print("Energy阈值校准 - 从验证集")
    print("=" * 80)
    
    # 参数配置
    MODEL_PATH = "./best_intent_model"
    VALIDATION_CSV = "validation_set.csv"  # 需要您提供验证集CSV
    TEXT_COL = "text"
    LABEL_COL = "label"
    BATCH_SIZE = 64  # 2万条数据可以用较大的batch_size加速
    
    # 检查文件是否存在
    if not Path(VALIDATION_CSV).exists():
        print(f"\n⚠ 错误：找不到验证集文件 {VALIDATION_CSV}")
        print("\n请确保您有以下任一文件：")
        print("  1. validation_set.csv - 验证集")
        print("  2. 或在训练脚本中保存验证集")
        print("\n如果您还没有保存验证集，可以修改xz_bert.py在划分数据时保存：")
        print("  val_df.to_csv('validation_set.csv', index=False)")
        return
    
    # 加载验证集
    print(f"\n加载验证集: {VALIDATION_CSV}")
    df_val = pd.read_csv(VALIDATION_CSV)
    print(f"验证集大小: {len(df_val)} 条")
    
    # 初始化校准器
    calibrator = EnergyThresholdCalibrator(model_path=MODEL_PATH, temperature=1.0)
    
    # 校准阈值
    result = calibrator.calibrate_from_dataframe(
        df=df_val,
        text_col=TEXT_COL,
        label_col=LABEL_COL,
        batch_size=BATCH_SIZE
    )
    
    # 保存结果
    calibrator.save_calibration_result(result, output_path="energy_thresholds.json")
    
    # 打印使用建议
    print("\n" + "=" * 80)
    print("使用建议")
    print("=" * 80)
    print("\n推荐使用MODERATE（适中）策略:")
    moderate = result['recommendations']['moderate']
    print(f"""
在 xz_bert_inference_energy.py 中设置：

classifier = EnergyBasedIntentClassifier(
    model_path="./best_intent_model",
    energy_threshold_low={moderate['threshold_low']:.4f},
    energy_threshold_high={moderate['threshold_high']:.4f},
    temperature=1.0
)
""")


def main_calibrate_with_ood_samples():
    """
    主函数：使用真实OOD样本校准阈值
    适用场景：您收集了一批真实的OOD样本（如"今天星期几"等）
    """
    print("=" * 80)
    print("Energy阈值校准 - 带OOD样本")
    print("=" * 80)
    
    MODEL_PATH = "./best_intent_model"
    
    # 方案1：从CSV加载
    ID_CSV = "validation_set.csv"    # ID样本（验证集）
    OOD_CSV = "ood_samples.csv"      # OOD样本（需要您收集）
    
    if Path(ID_CSV).exists() and Path(OOD_CSV).exists():
        print(f"\n从CSV加载数据...")
        df_id = pd.read_csv(ID_CSV)
        df_ood = pd.read_csv(OOD_CSV)
        
        id_texts = df_id['text'].tolist()
        ood_texts = df_ood['text'].tolist()
    else:
        # 方案2：手动定义一些OOD样本（用于快速测试）
        print(f"\n⚠ 找不到CSV文件，使用内置测试样本")
        print("提示：您可以创建ood_samples.csv来提供真实OOD样本")
        
        # 需要您提供验证集的ID样本
        if Path(ID_CSV).exists():
            df_id = pd.read_csv(ID_CSV)
            id_texts = df_id['text'].tolist()
        else:
            print(f"⚠ 错误：找不到ID样本文件 {ID_CSV}")
            return
        
        # 一些典型的OOD样本（您应该根据实际bad case扩充）
        ood_texts = [
            "今天星期几",
            "明天天气怎么样",
            "推荐一部好看的电影",
            "区块链技术是什么",
            "量子计算机的工作原理",
            "帮我写一篇作文",
            "如何做一道红烧肉",
            "最近股市怎么样",
            "Python怎么学",
            "你好",
            "谢谢",
            "再见",
            "春天在哪里",
            "给我讲个笑话",
            "1+1等于几",
        ]
        print(f"使用 {len(ood_texts)} 个内置OOD样本")
    
    # 初始化校准器
    calibrator = EnergyThresholdCalibrator(model_path=MODEL_PATH, temperature=1.0)
    
    # 校准阈值
    result = calibrator.calibrate_with_ood_samples(
        id_texts=id_texts,
        ood_texts=ood_texts,
        batch_size=64
    )
    
    # 保存结果
    calibrator.save_calibration_result(result, output_path="energy_thresholds_with_ood.json")
    
    # 打印使用建议
    print("\n" + "=" * 80)
    print("使用建议")
    print("=" * 80)
    best_threshold = result['best_threshold']['value']
    print(f"""
推荐使用最佳阈值:

classifier = EnergyBasedIntentClassifier(
    model_path="./best_intent_model",
    energy_threshold_low=0.0,  # 或设为ID均值的某个分位数
    energy_threshold_high={best_threshold:.4f},
    temperature=1.0
)

预期效果：
- ID样本接受率: {result['best_threshold']['tpr']:.1%}
- OOD样本拒绝率: {result['best_threshold']['tnr']:.1%}
""")


if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--with-ood":
        # 使用真实OOD样本校准
        main_calibrate_with_ood_samples()
    else:
        # 从验证集校准（默认）
        main_calibrate_from_validation_set()
    
    print("\n" + "=" * 80)
    print("校准完成！")
    print("=" * 80)









