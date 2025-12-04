"""
将标注数据转换为训练格式

输入: data/sampled_for_annotation.csv（已标注）
输出: data/intent_data_llm.csv（训练格式）
"""

import pandas as pd
from pathlib import Path

def convert_to_training_format(
    input_file: str = "data/sampled_for_annotation.csv",
    output_file: str = "data/intent_data_llm.csv",
    label_mapping: dict = None
):
    """
    转换标注数据为训练格式
    
    参数:
    - input_file: 标注后的数据文件
    - output_file: 输出的训练文件
    - label_mapping: 标签映射字典
    """
    
    print("=" * 70)
    print("转换标注数据为训练格式")
    print("=" * 70)
    
    # 默认标签映射
    if label_mapping is None:
        label_mapping = {
            '寿险相关': '寿险相关',
            '拒识': '其他',
            '其他': '其他',
        }
    
    # 读取数据
    print(f"\n加载数据: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"✓ 加载完成: {len(df)} 条")
    
    # 检查必要列
    if 'text' not in df.columns or 'label' not in df.columns:
        print("错误: 缺少必要列（text 或 label）")
        return
    
    # 统计标注情况
    n_total = len(df)
    n_labeled = df['label'].notna().sum()
    n_unlabeled = n_total - n_labeled
    
    print(f"\n标注统计:")
    print(f"  总数: {n_total}")
    print(f"  已标注: {n_labeled} ({n_labeled/n_total*100:.1f}%)")
    print(f"  未标注: {n_unlabeled} ({n_unlabeled/n_total*100:.1f}%)")
    
    if n_unlabeled > 0:
        print(f"\n⚠️ 警告: 还有 {n_unlabeled} 条未标注，将被过滤掉")
    
    # 过滤未标注的数据
    df = df[df['label'].notna()].copy()
    
    # 打印标签分布
    print(f"\n标签分布:")
    for label, count in df['label'].value_counts().items():
        print(f"  {label}: {count} ({count/len(df)*100:.1f}%)")
    
    # 映射标签
    print(f"\n应用标签映射:")
    for old_label, new_label in label_mapping.items():
        print(f"  '{old_label}' → '{new_label}'")
    
    df['mapped_label'] = df['label'].map(label_mapping)
    
    # 检查是否有未映射的标签
    unmapped = df[df['mapped_label'].isna()]
    if len(unmapped) > 0:
        print(f"\n⚠️ 警告: {len(unmapped)} 条数据的标签无法映射，将被过滤")
        print(f"未映射的标签: {unmapped['label'].unique().tolist()}")
        df = df[df['mapped_label'].notna()]
    
    # 转换为训练格式
    print(f"\n转换为训练格式...")
    
    # 格式: question | answer（用竖线分隔）
    training_df = pd.DataFrame({
        'question': df['text'],
        'answer': df['mapped_label']
    })
    
    # 打印最终统计
    print(f"\n最终训练数据:")
    print(f"  总数: {len(training_df)}")
    print(f"  类别分布:")
    for label, count in training_df['answer'].value_counts().items():
        print(f"    {label}: {count} ({count/len(training_df)*100:.1f}%)")
    
    # 数据质量检查
    print(f"\n数据质量检查:")
    
    # 1. 空值检查
    n_null = training_df.isnull().sum().sum()
    if n_null > 0:
        print(f"  ⚠️ 发现 {n_null} 个空值")
    else:
        print(f"  ✓ 无空值")
    
    # 2. 长度检查
    training_df['text_len'] = training_df['question'].str.len()
    print(f"  ✓ 文本长度: min={training_df['text_len'].min()}, max={training_df['text_len'].max()}, mean={training_df['text_len'].mean():.1f}")
    
    # 3. 重复检查
    n_duplicates = training_df['question'].duplicated().sum()
    if n_duplicates > 0:
        print(f"  ⚠️ 发现 {n_duplicates} 个重复样本")
        # 去重
        training_df = training_df.drop_duplicates(subset=['question'], keep='first')
        print(f"  ✓ 已去重，剩余 {len(training_df)} 条")
    else:
        print(f"  ✓ 无重复")
    
    # 保存
    print(f"\n保存训练数据...")
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 使用竖线分隔符（适配train_1.7b_classification.py）
    training_df[['question', 'answer']].to_csv(
        output_file,
        index=False,
        sep='|',
        encoding='utf-8-sig'
    )
    
    print(f"✓ 已保存到: {output_file}")
    
    # 生成数据预览
    print(f"\n数据预览（前5条）:")
    print("-" * 70)
    for i, row in training_df.head(5).iterrows():
        print(f"{i+1}. 问题: {row['question']}")
        print(f"   答案: {row['answer']}")
        print()
    print("-" * 70)
    
    # 生成训练数据划分建议
    print(f"\n训练数据划分建议:")
    n_total = len(training_df)
    n_train = int(n_total * 0.85)
    n_val = n_total - n_train
    
    print(f"  总数: {n_total}")
    print(f"  训练集: {n_train} (85%)")
    print(f"  验证集: {n_val} (15%)")
    
    print(f"\n类别平衡检查:")
    for label in training_df['answer'].unique():
        n_label = (training_df['answer'] == label).sum()
        ratio = n_label / n_total
        
        if 0.3 <= ratio <= 0.7:
            status = "✓ 平衡"
        elif 0.2 <= ratio <= 0.8:
            status = "~ 可接受"
        else:
            status = "⚠️ 不平衡"
        
        print(f"  {label}: {n_label} ({ratio*100:.1f}%) {status}")
    
    # 生成下一步指令
    print(f"\n" + "=" * 70)
    print(f"✓ 转换完成！")
    print(f"=" * 70)
    print(f"\n下一步: 开始训练模型")
    print(f"\n使用1.7B模型训练（推荐）:")
    print(f"  python train_1.7b_classification.py")
    print(f"\n或使用0.5B模型（更快）:")
    print(f"  python xz_qwen0.5b.py")
    print(f"\n训练预计时间:")
    print(f"  - 0.5B模型: 约30-60分钟（3个epoch）")
    print(f"  - 1.7B模型: 约50-75分钟（5个epoch）")
    print("=" * 70)


def validate_annotation_quality(input_file: str = "data/sampled_for_annotation.csv"):
    """
    验证标注质量
    
    检查:
    1. 标注完成度
    2. 标签一致性
    3. 可疑样本
    """
    
    print("\n" + "=" * 70)
    print("标注质量验证")
    print("=" * 70)
    
    df = pd.read_csv(input_file)
    
    # 1. 完成度检查
    n_total = len(df)
    n_labeled = df['label'].notna().sum()
    
    print(f"\n1. 完成度检查")
    print(f"   总数: {n_total}")
    print(f"   已标注: {n_labeled} ({n_labeled/n_total*100:.1f}%)")
    
    if n_labeled < n_total * 0.95:
        print(f"   ⚠️ 标注完成度 < 95%，建议补充标注")
    else:
        print(f"   ✓ 标注完成度良好")
    
    # 2. 标签分布
    print(f"\n2. 标签分布")
    labeled_df = df[df['label'].notna()]
    
    for label, count in labeled_df['label'].value_counts().items():
        ratio = count / len(labeled_df)
        print(f"   {label}: {count} ({ratio*100:.1f}%)")
        
        if ratio < 0.1:
            print(f"      ⚠️ 样本数过少，建议补充")
        elif ratio > 0.9:
            print(f"      ⚠️ 样本数过多，类别不平衡")
    
    # 3. 可疑样本检测
    print(f"\n3. 可疑样本检测")
    
    # 3.1 短文本 + 寿险相关
    if 'quality_score' in df.columns and 'label' in df.columns:
        suspicious = df[
            (df['text'].str.len() < 5) & 
            (df['label'] == '寿险相关')
        ]
        
        if len(suspicious) > 0:
            print(f"   ⚠️ 发现 {len(suspicious)} 个短文本被标为'寿险相关'")
            print(f"   示例: {suspicious['text'].head(3).tolist()}")
        else:
            print(f"   ✓ 未发现短文本异常标注")
    
    # 3.2 低质量得分 + 寿险相关
    if 'quality_score' in df.columns and 'label' in df.columns:
        suspicious = df[
            (df['quality_score'] < 0.3) & 
            (df['label'] == '寿险相关')
        ]
        
        if len(suspicious) > 0:
            print(f"   ⚠️ 发现 {len(suspicious)} 个低质量样本被标为'寿险相关'")
            print(f"   示例: {suspicious['text'].head(3).tolist()}")
        else:
            print(f"   ✓ 未发现低质量异常标注")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="转换标注数据为训练格式")
    parser.add_argument(
        "--input",
        type=str,
        default="data/sampled_for_annotation.csv",
        help="标注后的数据文件"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/intent_data_llm.csv",
        help="输出的训练文件"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="验证标注质量"
    )
    
    args = parser.parse_args()
    
    # 验证标注质量（可选）
    if args.validate:
        validate_annotation_quality(args.input)
    
    # 转换格式
    convert_to_training_format(
        input_file=args.input,
        output_file=args.output
    )

