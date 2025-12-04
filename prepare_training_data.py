"""
准备模型训练、验证和测试数据集
"""
import pandas as pd
from pathlib import Path
from typing import List


def merge_excel_files(file_list: List[str], output_file: str = None) -> pd.DataFrame:
    """
    合并多个Excel文件
    
    参数:
        file_list: Excel文件路径列表
        output_file: 输出文件路径，如果为None则不保存
    
    返回:
        合并后的DataFrame
    """
    print(f"\n合并 {len(file_list)} 个文件:")
    
    dfs = []
    for file_path in file_list:
        if not Path(file_path).exists():
            print(f"  警告: 文件不存在，跳过 - {file_path}")
            continue
        
        df = pd.read_excel(file_path)
        print(f"  - {file_path}: {len(df)} 行")
        dfs.append(df)
    
    if not dfs:
        print("  错误: 没有找到任何有效文件")
        return pd.DataFrame()
    
    # 合并所有DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"合并后总行数: {len(merged_df)}")
    
    # 如果指定了输出文件，则保存
    if output_file:
        merged_df.to_excel(output_file, index=False)
        print(f"已保存到: {output_file}")
    
    return merged_df


def sample_by_ratio(df: pd.DataFrame, sample_size: int, label_column: str = '模型输出assistant') -> pd.DataFrame:
    """
    按照标签比例抽取样本
    
    参数:
        df: 输入DataFrame
        sample_size: 要抽取的样本数量
        label_column: 标签列名
    
    返回:
        抽取后的DataFrame
    """
    if len(df) < sample_size:
        print(f"警告: 数据总量({len(df)})少于需要抽取的数量({sample_size})，返回全部数据")
        return df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 统计各类别的数量和比例
    label_counts = df[label_column].value_counts()
    print(f"\n原始数据分布:")
    for label, count in label_counts.items():
        ratio = count / len(df) * 100
        print(f"  {label}: {count} ({ratio:.2f}%)")
    
    # 按比例计算每个类别需要抽取的数量
    sampled_dfs = []
    for label, count in label_counts.items():
        ratio = count / len(df)
        target_count = int(sample_size * ratio)
        
        # 如果某个类别的数据不足，则全部抽取
        label_df = df[df[label_column] == label]
        if len(label_df) < target_count:
            print(f"  警告: {label}类别数据不足，全部抽取({len(label_df)}个)")
            sampled_dfs.append(label_df)
        else:
            sampled = label_df.sample(n=target_count, random_state=42)
            sampled_dfs.append(sampled)
            print(f"  从{label}抽取: {target_count} 个")
    
    # 合并抽取的数据
    result = pd.concat(sampled_dfs, ignore_index=True)
    
    # 打乱顺序
    result = result.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n抽取后数据分布:")
    label_counts = result[label_column].value_counts()
    for label, count in label_counts.items():
        ratio = count / len(result) * 100
        print(f"  {label}: {count} ({ratio:.2f}%)")
    print(f"总计: {len(result)} 行")
    
    return result


def get_remaining_data(original_df: pd.DataFrame, sampled_df: pd.DataFrame, 
                       key_column: str = '模型输入human') -> pd.DataFrame:
    """
    获取抽取后剩余的数据
    
    参数:
        original_df: 原始DataFrame
        sampled_df: 已抽取的DataFrame
        key_column: 用于判断重复的列名
    
    返回:
        剩余的DataFrame
    """
    # 使用内容来判断，而不是索引
    # 创建已抽取数据的集合（包含行的完整内容作为元组）
    sampled_set = set()
    for _, row in sampled_df.iterrows():
        # 将整行转换为元组，用于判断重复
        row_tuple = tuple(row.values)
        sampled_set.add(row_tuple)
    
    # 找出未被抽取的数据
    remaining_rows = []
    for idx, row in original_df.iterrows():
        row_tuple = tuple(row.values)
        if row_tuple not in sampled_set:
            remaining_rows.append(row)
    
    remaining_df = pd.DataFrame(remaining_rows)
    
    print(f"剩余数据: {len(remaining_df)} 行")
    print(f"验证: 原始数据 {len(original_df)} = 抽取数据 {len(sampled_df)} + 剩余数据 {len(remaining_df)}")
    
    if len(original_df) != len(sampled_df) + len(remaining_df):
        print("警告: 数据总数不匹配，可能存在重复数据")
    
    return remaining_df


def validate_no_duplicates(train_file: str, val_file: str, test_file: str, 
                          key_column: str = '模型输入human'):
    """
    验证训练集、验证集和测试集之间没有重复数据
    
    参数:
        train_file: 训练集文件路径
        val_file: 验证集文件路径
        test_file: 测试集文件路径
        key_column: 用于判断重复的关键列
    
    返回:
        bool: True表示没有重复，False表示有重复
    """
    print("\n" + "=" * 60)
    print("交叉验证: 检查数据集之间是否有重复")
    print("=" * 60)
    
    # 读取三个文件
    train_df = pd.read_excel(train_file)
    val_df = pd.read_excel(val_file)
    test_df = pd.read_excel(test_file)
    
    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_df)} 行")
    print(f"  验证集: {len(val_df)} 行")
    print(f"  测试集: {len(test_df)} 行")
    print(f"  总计: {len(train_df) + len(val_df) + len(test_df)} 行")
    
    # 将每行转换为元组用于比较
    def df_to_set(df):
        return set(tuple(row) for row in df.values)
    
    train_set = df_to_set(train_df)
    val_set = df_to_set(val_df)
    test_set = df_to_set(test_df)
    
    print(f"\n去重后的唯一数据:")
    print(f"  训练集: {len(train_set)} 行")
    print(f"  验证集: {len(val_set)} 行")
    print(f"  测试集: {len(test_set)} 行")
    
    # 检查内部重复
    has_internal_dup = False
    if len(train_set) < len(train_df):
        print(f"\n⚠️  训练集内部有 {len(train_df) - len(train_set)} 条重复数据")
        has_internal_dup = True
    if len(val_set) < len(val_df):
        print(f"⚠️  验证集内部有 {len(val_df) - len(val_set)} 条重复数据")
        has_internal_dup = True
    if len(test_set) < len(test_df):
        print(f"⚠️  测试集内部有 {len(test_df) - len(test_set)} 条重复数据")
        has_internal_dup = True
    
    # 检查数据集之间的重复
    print(f"\n检查数据集之间的重复:")
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    has_cross_dup = False
    
    if train_val_overlap:
        print(f"❌ 训练集与验证集有 {len(train_val_overlap)} 条重复数据!")
        has_cross_dup = True
        # 显示前几条重复数据
        for i, row_tuple in enumerate(list(train_val_overlap)[:3]):
            print(f"   示例{i+1}: {row_tuple}")
    else:
        print(f"✅ 训练集与验证集: 无重复")
    
    if train_test_overlap:
        print(f"❌ 训练集与测试集有 {len(train_test_overlap)} 条重复数据!")
        has_cross_dup = True
        # 显示前几条重复数据
        for i, row_tuple in enumerate(list(train_test_overlap)[:3]):
            print(f"   示例{i+1}: {row_tuple}")
    else:
        print(f"✅ 训练集与测试集: 无重复")
    
    if val_test_overlap:
        print(f"❌ 验证集与测试集有 {len(val_test_overlap)} 条重复数据!")
        has_cross_dup = True
        # 显示前几条重复数据
        for i, row_tuple in enumerate(list(val_test_overlap)[:3]):
            print(f"   示例{i+1}: {row_tuple}")
    else:
        print(f"✅ 验证集与测试集: 无重复")
    
    # 总结
    print("\n" + "=" * 60)
    if not has_internal_dup and not has_cross_dup:
        print("✅ 验证通过: 所有数据集都没有重复数据!")
        print("=" * 60)
        return True
    else:
        print("❌ 验证失败: 发现重复数据，请检查!")
        print("=" * 60)
        return False


def prepare_training_data(
    input_files: List[str],
    test_size: int = 1500,
    val_size: int = 1500,
    output_dir: str = "prepared_data"
):
    """
    准备训练、验证和测试数据集
    
    参数:
        input_files: 5个输入文件的路径列表
        test_size: 测试集大小
        val_size: 验证集大小
        output_dir: 输出目录
    """
    if len(input_files) != 5:
        print(f"错误: 需要提供5个文件，当前提供了{len(input_files)}个")
        return
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("步骤1: 合并前三个文件作为初始训练集")
    print("=" * 60)
    train_partial_file = output_path / "train_partial.xlsx"
    train_partial_df = merge_excel_files(
        input_files[:3],
        output_file=str(train_partial_file)
    )
    
    print("\n" + "=" * 60)
    print("步骤2: 合并后两个文件")
    print("=" * 60)
    combined_last_two = merge_excel_files(input_files[3:5])
    
    print("\n" + "=" * 60)
    print(f"步骤3: 按比例抽取测试集和验证集 (总计 {test_size + val_size} 个样本)")
    print("=" * 60)
    total_sample_size = test_size + val_size
    sampled_data = sample_by_ratio(combined_last_two, total_sample_size)
    
    # 从抽取的数据中分割测试集和验证集
    print(f"\n将抽取的数据分为测试集({test_size})和验证集({val_size})")
    
    # 对每个类别分别抽取，保持比例
    label_column = '模型输出assistant'
    test_dfs = []
    val_dfs = []
    
    for label in sampled_data[label_column].unique():
        label_data = sampled_data[sampled_data[label_column] == label]
        label_count = len(label_data)
        
        # 计算该类别在测试集中的数量
        test_count = int(label_count * test_size / total_sample_size)
        
        # 打乱并分割
        label_data_shuffled = label_data.sample(frac=1, random_state=42).reset_index(drop=True)
        test_dfs.append(label_data_shuffled[:test_count])
        val_dfs.append(label_data_shuffled[test_count:])
    
    test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 保存测试集和验证集
    test_file = output_path / "test.xlsx"
    val_file = output_path / "validation.xlsx"
    
    test_df.to_excel(test_file, index=False)
    print(f"\n测试集已保存: {test_file} ({len(test_df)} 行)")
    
    val_df.to_excel(val_file, index=False)
    print(f"验证集已保存: {val_file} ({len(val_df)} 行)")
    
    print("\n" + "=" * 60)
    print("步骤4: 合并剩余数据与初始训练集，生成完整训练集")
    print("=" * 60)
    
    # 获取剩余数据
    all_sampled = pd.concat([test_df, val_df], ignore_index=True)
    remaining_df = get_remaining_data(combined_last_two, all_sampled)
    
    # 合并剩余数据和初始训练集
    train_full_df = pd.concat([train_partial_df, remaining_df], ignore_index=True)
    
    # 打乱训练集
    train_full_df = train_full_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    train_full_file = output_path / "train_full.xlsx"
    train_full_df.to_excel(train_full_file, index=False)
    print(f"\n完整训练集已保存: {train_full_file} ({len(train_full_df)} 行)")
    
    print("\n" + "=" * 60)
    print("完成! 数据集统计:")
    print("=" * 60)
    print(f"训练集: {len(train_full_df)} 行 -> {train_full_file}")
    print(f"验证集: {len(val_df)} 行 -> {val_file}")
    print(f"测试集: {len(test_df)} 行 -> {test_file}")
    print(f"总计: {len(train_full_df) + len(val_df) + len(test_df)} 行")
    
    # 显示各数据集的标签分布
    print("\n训练集标签分布:")
    for label, count in train_full_df[label_column].value_counts().items():
        print(f"  {label}: {count}")
    
    print("\n验证集标签分布:")
    for label, count in val_df[label_column].value_counts().items():
        print(f"  {label}: {count}")
    
    print("\n测试集标签分布:")
    for label, count in test_df[label_column].value_counts().items():
        print(f"  {label}: {count}")
    
    # 自动执行交叉验证
    validate_no_duplicates(
        train_file=str(train_full_file),
        val_file=str(val_file),
        test_file=str(test_file)
    )


if __name__ == "__main__":
    # ========== 配置区域 ==========
    # 5个输入文件路径
    INPUT_FILES = [
        "file1.xlsx",  # 文件1
        "file2.xlsx",  # 文件2
        "file3.xlsx",  # 文件3
        "file4.xlsx",  # 文件4
        "file5.xlsx",  # 文件5
    ]
    
    # 测试集和验证集大小
    TEST_SIZE = 1500      # 测试集样本数
    VAL_SIZE = 1500       # 验证集样本数
    
    # 输出目录
    OUTPUT_DIR = "prepared_data"
    # ==============================
    
    # 检查所有文件是否存在
    missing_files = [f for f in INPUT_FILES if not Path(f).exists()]
    if missing_files:
        print("错误: 以下文件不存在:")
        for f in missing_files:
            print(f"  - {f}")
        print("\n请修改脚本中的 INPUT_FILES 变量，指定正确的文件路径")
    else:
        # 执行数据准备
        prepare_training_data(
            input_files=INPUT_FILES,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
            output_dir=OUTPUT_DIR
        )

