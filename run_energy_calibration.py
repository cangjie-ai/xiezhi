# 一键运行Energy阈值校准和测试
# 使用步骤：
# 1. 先训练模型并保存验证集
# 2. 运行此脚本校准阈值
# 3. 使用校准后的阈值进行推理

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

def step1_prepare_validation_set():
    """
    步骤1：准备验证集
    如果您在训练时没有保存验证集，运行此函数从原始数据重新划分
    """
    print("=" * 80)
    print("步骤1：准备验证集")
    print("=" * 80)
    
    # 检查是否已有验证集
    if Path("validation_set.csv").exists():
        print("✓ 验证集已存在: validation_set.csv")
        df_val = pd.read_csv("validation_set.csv")
        print(f"  大小: {len(df_val)} 条")
        print(f"  标签分布:\n{df_val['label'].value_counts()}")
        return True
    
    # 如果没有，从原始数据重新划分
    print("验证集不存在，从原始数据重新划分...")
    
    ORIGINAL_DATA = "data/intent_data_label.csv"
    if not Path(ORIGINAL_DATA).exists():
        print(f"⚠ 错误：找不到原始数据 {ORIGINAL_DATA}")
        print("请修改ORIGINAL_DATA变量指向您的数据文件")
        return False
    
    # 加载原始数据
    df = pd.read_csv(ORIGINAL_DATA)
    print(f"原始数据大小: {len(df)} 条")
    
    # 数据预处理（与xz_bert.py保持一致）
    def prepare_text_for_bert(row):
        system_col = [col for col in df.columns if 'system' in col.lower()][0]
        human_col = [col for col in df.columns if 'human' in col.lower()][0]
        system_text = str(row[system_col]) if pd.notna(row[system_col]) else ""
        human_text = str(row[human_col]) if pd.notna(row[human_col]) else ""
        if system_text:
            return f"{system_text} {human_text}"
        else:
            return human_text
    
    df['text'] = df.apply(prepare_text_for_bert, axis=1)
    
    # 提取标签（二分类版本）
    answer_col = [col for col in df.columns if 'answer' in col.lower() or '答案' in col][0]
    
    def extract_label(answer):
        answer_str = str(answer).lower()
        if '寿险' in answer_str or 'life insurance' in answer_str or '定期寿' in answer_str or '终身寿' in answer_str:
            return 1  # 寿险相关
        return 0  # 拒识
    
    df['label'] = df[answer_col].apply(extract_label)
    df_processed = df[['text', 'label']]
    df_processed = df_processed[df_processed['text'].str.strip() != '']
    
    # 划分训练集和验证集（与训练时保持一致）
    train_val_df, test_df = train_test_split(
        df_processed, test_size=0.1, random_state=42, stratify=df_processed['label']
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=0.15, random_state=42, stratify=train_val_df['label']
    )
    
    # 保存验证集
    val_df.to_csv('validation_set.csv', index=False)
    print(f"✓ 验证集已保存: validation_set.csv ({len(val_df)} 条)")
    print(f"  标签分布:\n{val_df['label'].value_counts()}")
    
    return True


def step2_calibrate_thresholds():
    """
    步骤2：校准Energy阈值
    """
    print("\n" + "=" * 80)
    print("步骤2：校准Energy阈值")
    print("=" * 80)
    
    from xz_bert_calibrate_energy import EnergyThresholdCalibrator
    
    MODEL_PATH = "./best_intent_model"
    if not Path(MODEL_PATH).exists():
        print(f"⚠ 错误：找不到模型 {MODEL_PATH}")
        print("请先运行 xz_bert.py 训练模型")
        return None
    
    # 加载验证集
    df_val = pd.read_csv("validation_set.csv")
    
    # 初始化校准器
    calibrator = EnergyThresholdCalibrator(model_path=MODEL_PATH, temperature=1.0)
    
    # 校准阈值
    result = calibrator.calibrate_from_dataframe(
        df=df_val,
        text_col="text",
        label_col="label",
        batch_size=64
    )
    
    # 保存结果
    calibrator.save_calibration_result(result, output_path="energy_thresholds.json")
    
    return result


def step3_test_with_calibrated_thresholds():
    """
    步骤3：使用校准后的阈值测试
    """
    print("\n" + "=" * 80)
    print("步骤3：使用校准后的阈值测试")
    print("=" * 80)
    
    # 加载校准结果
    if not Path("energy_thresholds.json").exists():
        print("⚠ 错误：找不到校准结果，请先运行步骤2")
        return
    
    with open("energy_thresholds.json", 'r', encoding='utf-8') as f:
        calibration = json.load(f)
    
    # 使用适中策略的阈值
    moderate = calibration['recommendations']['moderate']
    threshold_low = moderate['threshold_low']
    threshold_high = moderate['threshold_high']
    
    print(f"使用阈值: low={threshold_low:.4f}, high={threshold_high:.4f}")
    
    # 初始化分类器
    from xz_bert_inference_energy import EnergyBasedIntentClassifier
    
    classifier = EnergyBasedIntentClassifier(
        model_path="./best_intent_model",
        energy_threshold_low=threshold_low,
        energy_threshold_high=threshold_high,
        temperature=1.0
    )
    
    # 测试样例（包含ID和OOD）
    test_cases = [
        # ID样本 - 寿险相关
        ("我想了解一下定期寿险产品", "预期: 寿险相关"),
        ("寿险保费怎么计算的？", "预期: 寿险相关"),
        ("终身寿险和定期寿险有什么区别？", "预期: 寿险相关"),
        
        # ID样本 - 拒识
        ("你们公司的重疾险怎么样？", "预期: 拒识"),
        ("我要买车险", "预期: 拒识"),
        
        # OOD样本 - 应该被检测出来
        ("今天星期几", "预期: OOD"),
        ("明天天气怎么样", "预期: OOD"),
        ("区块链技术介绍", "预期: OOD"),
        ("帮我写一篇作文", "预期: OOD"),
        ("量子计算机的工作原理", "预期: OOD"),
        ("推荐一部好看的电影", "预期: OOD"),
        ("你好", "预期: OOD"),
        ("1+1等于几", "预期: OOD"),
    ]
    
    print("\n测试结果：")
    print("-" * 80)
    
    # 统计结果
    ood_detected = 0
    total_ood = 0
    
    for text, expected in test_cases:
        label, confidence, details = classifier.predict_single(text, return_details=True)
        
        is_ood = "OOD" in expected
        is_detected_as_ood = "OOD" in label
        
        if is_ood:
            total_ood += 1
            if is_detected_as_ood:
                ood_detected += 1
        
        # 标记是否符合预期
        status = "✓" if (is_ood == is_detected_as_ood) else "✗"
        
        print(f"\n{status} 输入: {text}")
        print(f"  {expected}")
        print(f"  结果: {label} (置信度: {confidence})")
        print(f"  Energy: {details['energy']:.4f}")
        print(f"  原因: {details['reason']}")
    
    print("\n" + "-" * 80)
    print(f"OOD检测率: {ood_detected}/{total_ood} = {ood_detected/total_ood:.1%}")
    print("-" * 80)


def main():
    """主流程"""
    print("=" * 80)
    print("Energy-based OOD检测 - 完整流程")
    print("=" * 80)
    
    # 步骤1：准备验证集
    if not step1_prepare_validation_set():
        print("\n⚠ 步骤1失败，流程终止")
        return
    
    # 步骤2：校准阈值
    result = step2_calibrate_thresholds()
    if result is None:
        print("\n⚠ 步骤2失败，流程终止")
        return
    
    # 步骤3：测试
    step3_test_with_calibrated_thresholds()
    
    # 打印最终建议
    print("\n" + "=" * 80)
    print("完成！后续使用建议")
    print("=" * 80)
    
    moderate = result['recommendations']['moderate']
    print(f"""
1. 在生产环境中使用校准后的阈值：

from xz_bert_inference_energy import EnergyBasedIntentClassifier

classifier = EnergyBasedIntentClassifier(
    model_path="./best_intent_model",
    energy_threshold_low={moderate['threshold_low']:.4f},
    energy_threshold_high={moderate['threshold_high']:.4f},
    temperature=1.0
)

label, confidence = classifier.predict_single("用户输入")

2. 对于confidence="medium"的样本，建议接入LLM进行二次验证

3. 定期收集bad cases（被误判的OOD样本），重新运行校准

4. 校准结果已保存在 energy_thresholds.json，可随时查看
""")


if __name__ == "__main__":
    main()









