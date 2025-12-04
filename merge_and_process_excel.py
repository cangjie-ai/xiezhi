"""
Excel文件合并和数据处理脚本

功能：
1. 合并多个结构相同的xlsx文件
2. 解析output列的JSON数据，拆分为thinking和label两列
3. 与另一个excel文件按text列合并（严格大小写匹配）

使用方法：
python merge_and_process_excel.py
"""

import pandas as pd
import json
import glob
import os
from pathlib import Path


def merge_excel_files(input_pattern, output_file=None):
    """
    合并多个结构相同的Excel文件
    
    Args:
        input_pattern: 文件匹配模式，例如 "data/*.xlsx" 或文件列表
        output_file: 可选的输出文件名
    
    Returns:
        合并后的DataFrame
    """
    # 如果是字符串模式，使用glob获取文件列表
    if isinstance(input_pattern, str):
        files = glob.glob(input_pattern)
    else:
        files = input_pattern
    
    if not files:
        raise ValueError(f"未找到匹配的文件: {input_pattern}")
    
    print(f"找到 {len(files)} 个文件待合并:")
    for f in files:
        print(f"  - {f}")
    
    # 读取所有Excel文件并合并
    dfs = []
    for file in files:
        try:
            df = pd.read_excel(file)
            dfs.append(df)
            print(f"✓ 成功读取: {file} ({len(df)} 行)")
        except Exception as e:
            print(f"✗ 读取失败: {file}, 错误: {e}")
    
    if not dfs:
        raise ValueError("没有成功读取任何文件")
    
    # 合并所有DataFrame
    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"\n合并完成，总共 {len(merged_df)} 行数据")
    
    # 可选：保存合并结果
    if output_file:
        merged_df.to_excel(output_file, index=False)
        print(f"合并结果已保存至: {output_file}")
    
    return merged_df


def parse_json_column(df, json_column='output', drop_original=True):
    """
    解析JSON格式的列，拆分为多个独立列
    
    Args:
        df: 输入DataFrame
        json_column: 包含JSON数据的列名
        drop_original: 是否删除原始列
    
    Returns:
        处理后的DataFrame
    """
    if json_column not in df.columns:
        raise ValueError(f"列 '{json_column}' 不存在于DataFrame中")
    
    print(f"\n开始解析 '{json_column}' 列...")
    
    thinking_list = []
    label_list = []
    error_count = 0
    
    for idx, value in enumerate(df[json_column]):
        try:
            # 尝试解析JSON
            if pd.isna(value):
                thinking_list.append(None)
                label_list.append(None)
            elif isinstance(value, str):
                parsed = json.loads(value)
                thinking_list.append(parsed.get('thinking', None))
                label_list.append(parsed.get('label', None))
            elif isinstance(value, dict):
                thinking_list.append(value.get('thinking', None))
                label_list.append(value.get('label', None))
            else:
                thinking_list.append(None)
                label_list.append(None)
                error_count += 1
        except (json.JSONDecodeError, Exception) as e:
            thinking_list.append(None)
            label_list.append(None)
            error_count += 1
            if error_count <= 5:  # 只打印前5个错误
                print(f"  警告: 第 {idx} 行解析失败: {e}")
    
    # 添加新列
    df['thinking'] = thinking_list
    df['label'] = label_list
    
    # 删除原始列
    if drop_original:
        df = df.drop(columns=[json_column])
    
    print(f"✓ 解析完成")
    print(f"  - thinking列: {df['thinking'].notna().sum()} 个有效值")
    print(f"  - label列: {df['label'].notna().sum()} 个有效值")
    if error_count > 0:
        print(f"  - 解析失败: {error_count} 行")
    
    return df


def merge_with_reference_excel(df, reference_file, merge_column='text', how='inner'):
    """
    与参考Excel文件按指定列合并（严格大小写匹配）
    
    Args:
        df: 主DataFrame
        reference_file: 参考Excel文件路径
        merge_column: 用于合并的列名
        how: 合并方式 ('inner', 'left', 'right', 'outer')
    
    Returns:
        合并后的DataFrame
    """
    print(f"\n读取参考文件: {reference_file}")
    reference_df = pd.read_excel(reference_file)
    print(f"参考文件有 {len(reference_df)} 行数据")
    
    # 确保合并列存在
    if merge_column not in df.columns:
        raise ValueError(f"主DataFrame中不存在列 '{merge_column}'")
    if merge_column not in reference_df.columns:
        raise ValueError(f"参考文件中不存在列 '{merge_column}'")
    
    # 确保合并列为字符串类型（严格大小写匹配）
    print(f"\n确保 '{merge_column}' 列为字符串类型...")
    df[merge_column] = df[merge_column].astype(str)
    reference_df[merge_column] = reference_df[merge_column].astype(str)
    
    # 检查重复值
    df_duplicates = df[merge_column].duplicated().sum()
    ref_duplicates = reference_df[merge_column].duplicated().sum()
    if df_duplicates > 0:
        print(f"  警告: 主DataFrame中有 {df_duplicates} 个重复的 {merge_column} 值")
    if ref_duplicates > 0:
        print(f"  警告: 参考文件中有 {ref_duplicates} 个重复的 {merge_column} 值")
    
    # 合并数据
    print(f"\n按 '{merge_column}' 列合并数据 (方式: {how})...")
    merged_df = pd.merge(df, reference_df, on=merge_column, how=how, suffixes=('', '_ref'))
    
    print(f"✓ 合并完成，结果有 {len(merged_df)} 行数据")
    
    return merged_df


def main():
    """主函数"""
    print("=" * 60)
    print("Excel文件合并和处理脚本")
    print("=" * 60)
    
    # ==================== 配置区 ====================
    # TODO: 修改以下路径为实际文件路径
    
    # 步骤1: 指定要合并的10个Excel文件
    # 方式1: 使用通配符匹配
    input_files = "data/file_*.xlsx"  # 例如: data/file_1.xlsx, data/file_2.xlsx, ...
    
    # 方式2: 明确指定文件列表（推荐）
    # input_files = [
    #     "data/file_1.xlsx",
    #     "data/file_2.xlsx",
    #     "data/file_3.xlsx",
    #     "data/file_4.xlsx",
    #     "data/file_5.xlsx",
    #     "data/file_6.xlsx",
    #     "data/file_7.xlsx",
    #     "data/file_8.xlsx",
    #     "data/file_9.xlsx",
    #     "data/file_10.xlsx",
    # ]
    
    # 步骤3: 指定参考Excel文件
    reference_file = "reference.xlsx"  # TODO: 修改为实际文件名
    
    # 输出文件
    intermediate_output = "merged_with_parsed_columns.xlsx"  # 中间结果
    final_output = "final_merged_result.xlsx"  # 最终结果
    
    # ==================== 处理流程 ====================
    
    try:
        # 步骤1: 合并多个Excel文件
        print("\n【步骤1】合并Excel文件...")
        df = merge_excel_files(input_files)
        
        # 步骤2: 解析output列的JSON数据
        print("\n【步骤2】解析JSON列...")
        df = parse_json_column(df, json_column='output', drop_original=True)
        
        # 保存中间结果
        df.to_excel(intermediate_output, index=False)
        print(f"\n中间结果已保存: {intermediate_output}")
        
        # 步骤3: 与参考文件合并
        print("\n【步骤3】与参考文件合并...")
        final_df = merge_with_reference_excel(
            df, 
            reference_file, 
            merge_column='text', 
            how='inner'  # 可选: 'inner', 'left', 'right', 'outer'
        )
        
        # 保存最终结果
        final_df.to_excel(final_output, index=False)
        print(f"\n✓ 最终结果已保存: {final_output}")
        
        # 显示结果摘要
        print("\n" + "=" * 60)
        print("处理完成！")
        print("=" * 60)
        print(f"最终数据集信息:")
        print(f"  - 行数: {len(final_df)}")
        print(f"  - 列数: {len(final_df.columns)}")
        print(f"  - 列名: {', '.join(final_df.columns)}")
        print(f"\n前5行预览:")
        print(final_df.head())
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

