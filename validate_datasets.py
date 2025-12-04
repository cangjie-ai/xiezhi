"""
验证训练集、验证集和测试集之间没有重复数据
可以单独运行此脚本对已生成的数据集进行验证
"""
import pandas as pd
from pathlib import Path


def validate_no_duplicates(train_file: str, val_file: str, test_file: str):
    """
    验证训练集、验证集和测试集之间没有重复数据
    
    参数:
        train_file: 训练集文件路径
        val_file: 验证集文件路径
        test_file: 测试集文件路径
    
    返回:
        bool: True表示没有重复，False表示有重复
    """
    print("\n" + "=" * 60)
    print("交叉验证: 检查数据集之间是否有重复")
    print("=" * 60)
    
    # 检查文件是否存在
    for file_path in [train_file, val_file, test_file]:
        if not Path(file_path).exists():
            print(f"❌ 错误: 文件不存在 - {file_path}")
            return False
    
    # 读取三个文件
    print(f"\n正在读取文件...")
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
    
    print(f"\n正在进行重复检查...")
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
        print(f"   前3条重复示例:")
        for i, row_tuple in enumerate(list(train_val_overlap)[:3]):
            print(f"   {i+1}. {row_tuple}")
    else:
        print(f"✅ 训练集与验证集: 无重复")
    
    if train_test_overlap:
        print(f"❌ 训练集与测试集有 {len(train_test_overlap)} 条重复数据!")
        has_cross_dup = True
        # 显示前几条重复数据
        print(f"   前3条重复示例:")
        for i, row_tuple in enumerate(list(train_test_overlap)[:3]):
            print(f"   {i+1}. {row_tuple}")
    else:
        print(f"✅ 训练集与测试集: 无重复")
    
    if val_test_overlap:
        print(f"❌ 验证集与测试集有 {len(val_test_overlap)} 条重复数据!")
        has_cross_dup = True
        # 显示前几条重复数据
        print(f"   前3条重复示例:")
        for i, row_tuple in enumerate(list(val_test_overlap)[:3]):
            print(f"   {i+1}. {row_tuple}")
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


if __name__ == "__main__":
    # ========== 配置区域 ==========
    # 指定要验证的三个文件路径
    TRAIN_FILE = "prepared_data/train_full.xlsx"
    VAL_FILE = "prepared_data/validation.xlsx"
    TEST_FILE = "prepared_data/test.xlsx"
    # ==============================
    
    # 执行验证
    result = validate_no_duplicates(
        train_file=TRAIN_FILE,
        val_file=VAL_FILE,
        test_file=TEST_FILE
    )
    
    if result:
        print("\n✅ 数据集可以安全用于模型训练!")
    else:
        print("\n❌ 请修复重复问题后再进行训练!")












