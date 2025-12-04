"""
合并和过滤CSV文件的脚本

功能：
1. 读取并合并2个格式一样的csv文件（以msg为key）
2. 合并第三个文件，只保留frequency <= 1的行
3. 以msg为key去重
4. 删除在第四个excel文件中出现过的msg
"""

import pandas as pd
import argparse
import sys
from pathlib import Path


def merge_and_filter_csv(
    csv_file1: str,
    csv_file2: str,
    csv_file3: str,
    excel_file4: str,
    output_file: str,
    msg_column: str = "msg",
    frequency_column: str = "frequency"
):
    """
    合并并过滤CSV文件
    
    Args:
        csv_file1: 第一个CSV文件路径
        csv_file2: 第二个CSV文件路径
        csv_file3: 第三个CSV文件路径（需要过滤frequency > 1的）
        excel_file4: 第四个Excel文件路径（用于排除msg）
        output_file: 输出文件路径
        msg_column: msg列的列名，默认为"msg"
        frequency_column: frequency列的列名，默认为"frequency"
    """
    
    print("=" * 60)
    print("开始处理文件...")
    print("=" * 60)
    
    # 步骤1: 读取并合并前两个CSV文件
    print(f"\n步骤1: 读取并合并前两个CSV文件")
    print(f"  - 读取文件1: {csv_file1}")
    df1 = pd.read_csv(csv_file1)
    print(f"    记录数: {len(df1)}")
    
    print(f"  - 读取文件2: {csv_file2}")
    df2 = pd.read_csv(csv_file2)
    print(f"    记录数: {len(df2)}")
    
    # 合并前两个文件
    merged_df = pd.concat([df1, df2], ignore_index=True)
    print(f"  - 合并后记录数: {len(merged_df)}")
    
    # 步骤2: 读取第三个文件并过滤frequency > 1的行
    print(f"\n步骤2: 读取第三个文件并过滤")
    print(f"  - 读取文件3: {csv_file3}")
    df3 = pd.read_csv(csv_file3)
    print(f"    原始记录数: {len(df3)}")
    
    # 检查是否有frequency列
    if frequency_column not in df3.columns:
        print(f"  警告: 文件3中没有找到'{frequency_column}'列，将包含所有记录")
        df3_filtered = df3
    else:
        # 只保留frequency <= 1的行
        df3_filtered = df3[df3[frequency_column] <= 1].copy()
        print(f"    过滤后记录数 (frequency <= 1): {len(df3_filtered)}")
        print(f"    删除的记录数 (frequency > 1): {len(df3) - len(df3_filtered)}")
    
    # 合并所有三个文件
    merged_df = pd.concat([merged_df, df3_filtered], ignore_index=True)
    print(f"  - 合并三个文件后总记录数: {len(merged_df)}")
    
    # 步骤3: 以msg为key去重
    print(f"\n步骤3: 以'{msg_column}'为key去重")
    
    # 检查是否有msg列
    if msg_column not in merged_df.columns:
        print(f"  错误: 找不到列'{msg_column}'")
        print(f"  可用的列: {list(merged_df.columns)}")
        sys.exit(1)
    
    before_dedup = len(merged_df)
    # 去重，保留第一次出现的记录
    merged_df = merged_df.drop_duplicates(subset=[msg_column], keep='first')
    after_dedup = len(merged_df)
    print(f"  - 去重前记录数: {before_dedup}")
    print(f"  - 去重后记录数: {after_dedup}")
    print(f"  - 删除的重复记录数: {before_dedup - after_dedup}")
    
    # 步骤4: 删除在第四个excel文件中出现过的msg
    print(f"\n步骤4: 删除在Excel文件中出现过的msg")
    print(f"  - 读取文件4: {excel_file4}")
    
    # 根据文件扩展名读取excel或csv
    file_ext = Path(excel_file4).suffix.lower()
    if file_ext in ['.xlsx', '.xls']:
        df4 = pd.read_excel(excel_file4)
    elif file_ext == '.csv':
        df4 = pd.read_csv(excel_file4)
    else:
        print(f"  错误: 不支持的文件格式 {file_ext}")
        sys.exit(1)
    
    print(f"    记录数: {len(df4)}")
    
    # 检查第四个文件是否有msg列
    if msg_column not in df4.columns:
        print(f"  警告: 文件4中没有找到'{msg_column}'列")
        print(f"  可用的列: {list(df4.columns)}")
        print(f"  将跳过此步骤")
        excluded_msgs = set()
    else:
        # 获取第四个文件中的所有msg
        excluded_msgs = set(df4[msg_column].dropna().unique())
        print(f"  - 文件4中唯一的msg数量: {len(excluded_msgs)}")
    
    # 过滤掉在第四个文件中出现过的msg
    before_filter = len(merged_df)
    merged_df = merged_df[~merged_df[msg_column].isin(excluded_msgs)]
    after_filter = len(merged_df)
    print(f"  - 过滤前记录数: {before_filter}")
    print(f"  - 过滤后记录数: {after_filter}")
    print(f"  - 删除的记录数: {before_filter - after_filter}")
    
    # 保存结果
    print(f"\n步骤5: 保存结果")
    merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"  - 输出文件: {output_file}")
    print(f"  - 最终记录数: {len(merged_df)}")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    
    return merged_df


def main():
    parser = argparse.ArgumentParser(
        description='合并和过滤CSV文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python merge_and_filter_csv.py file1.csv file2.csv file3.csv exclude.xlsx output.csv
  
  python merge_and_filter_csv.py file1.csv file2.csv file3.csv exclude.xlsx output.csv --msg-column message --frequency-column freq
        """
    )
    
    parser.add_argument('csv_file1', help='第一个CSV文件路径')
    parser.add_argument('csv_file2', help='第二个CSV文件路径')
    parser.add_argument('csv_file3', help='第三个CSV文件路径（会过滤frequency > 1的行）')
    parser.add_argument('excel_file4', help='第四个Excel文件路径（用于排除msg）')
    parser.add_argument('output_file', help='输出文件路径')
    parser.add_argument('--msg-column', default='msg', help='msg列的列名（默认: msg）')
    parser.add_argument('--frequency-column', default='frequency', help='frequency列的列名（默认: frequency）')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    for file_path in [args.csv_file1, args.csv_file2, args.csv_file3, args.excel_file4]:
        if not Path(file_path).exists():
            print(f"错误: 文件不存在: {file_path}")
            sys.exit(1)
    
    # 执行合并和过滤
    merge_and_filter_csv(
        csv_file1=args.csv_file1,
        csv_file2=args.csv_file2,
        csv_file3=args.csv_file3,
        excel_file4=args.excel_file4,
        output_file=args.output_file,
        msg_column=args.msg_column,
        frequency_column=args.frequency_column
    )


if __name__ == "__main__":
    main()









