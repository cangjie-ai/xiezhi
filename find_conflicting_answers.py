"""
查找多个Excel文件中问题相同但答案不同的冲突记录
"""
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict
import argparse


def normalize_text(text: str) -> str:
    """
    标准化文本：去除空格、换行、制表符等特殊字符
    
    Args:
        text: 原始文本
        
    Returns:
        标准化后的文本
    """
    if pd.isna(text):
        return ""
    
    # 转换为字符串
    text = str(text)
    
    # 去除所有空格、换行、制表符、回车等
    text = re.sub(r'\s+', '', text)
    
    # 转换为小写以便比较
    text = text.lower()
    
    return text


def load_excel_files(file_paths: List[str]) -> List[Dict]:
    """
    加载多个Excel文件
    
    Args:
        file_paths: Excel文件路径列表
        
    Returns:
        包含数据和元信息的字典列表
    """
    data_list = []
    
    for file_path in file_paths:
        try:
            df = pd.read_excel(file_path)
            
            # 检查是否包含必要的列
            if '问题' not in df.columns or 'answer' not in df.columns:
                print(f"警告: {file_path} 缺少'问题'或'answer'列，跳过此文件")
                continue
            
            # 添加来源列
            df['来源文件'] = Path(file_path).name
            
            data_list.append({
                'path': file_path,
                'name': Path(file_path).name,
                'data': df
            })
            
            print(f"成功加载: {file_path}, 共 {len(df)} 行")
            
        except Exception as e:
            print(f"错误: 无法加载 {file_path}: {e}")
    
    return data_list


def find_conflicts(data_list: List[Dict]) -> pd.DataFrame:
    """
    查找问题相同但答案不同的冲突记录
    
    Args:
        data_list: 包含数据的字典列表
        
    Returns:
        冲突记录的DataFrame
    """
    # 合并所有数据
    all_data = []
    
    for data_dict in data_list:
        df = data_dict['data'][['问题', 'answer', '来源文件']].copy()
        all_data.append(df)
    
    if not all_data:
        print("错误: 没有有效的数据")
        return pd.DataFrame()
    
    # 合并所有DataFrame
    combined_df = pd.concat(all_data, ignore_index=True)
    
    print(f"\n总共加载 {len(combined_df)} 条记录")
    
    # 添加标准化的问题列
    combined_df['问题_标准化'] = combined_df['问题'].apply(normalize_text)
    combined_df['answer_标准化'] = combined_df['answer'].apply(normalize_text)
    
    # 过滤掉空问题
    combined_df = combined_df[combined_df['问题_标准化'] != '']
    
    print(f"过滤空问题后剩余 {len(combined_df)} 条记录")
    
    # 按标准化的问题分组
    grouped = combined_df.groupby('问题_标准化')
    
    conflicts = []
    
    for question_normalized, group in grouped:
        # 获取唯一的答案（标准化后）
        unique_answers = group['answer_标准化'].unique()
        
        # 如果有多个不同的答案，说明存在冲突
        if len(unique_answers) > 1:
            # 对于每个唯一答案，获取一个代表记录
            answer_groups = group.groupby('answer_标准化')
            
            conflict_records = []
            for answer_normalized, answer_group in answer_groups:
                # 获取该答案的所有来源
                sources = answer_group['来源文件'].tolist()
                
                # 取第一条记录作为代表
                record = answer_group.iloc[0]
                
                conflict_records.append({
                    '问题': record['问题'],
                    'answer': record['answer'],
                    '来源文件': ', '.join(sources),
                    '出现次数': len(answer_group)
                })
            
            # 添加冲突组信息
            conflict_group = pd.DataFrame(conflict_records)
            conflict_group['冲突组ID'] = f"冲突_{len(conflicts) + 1}"
            conflict_group['该问题答案总数'] = len(conflict_records)
            
            conflicts.append(conflict_group)
    
    if not conflicts:
        print("\n未发现冲突记录")
        return pd.DataFrame()
    
    # 合并所有冲突
    conflicts_df = pd.concat(conflicts, ignore_index=True)
    
    # 重新排列列顺序
    columns_order = ['冲突组ID', '该问题答案总数', '问题', 'answer', '来源文件', '出现次数']
    conflicts_df = conflicts_df[columns_order]
    
    # 按冲突组ID排序
    conflicts_df = conflicts_df.sort_values('冲突组ID')
    
    print(f"\n发现 {len(conflicts)} 个问题存在冲突")
    print(f"总共 {len(conflicts_df)} 条冲突记录")
    
    return conflicts_df


def save_conflicts_to_excel(conflicts_df: pd.DataFrame, output_path: str):
    """
    将冲突记录保存到Excel文件
    
    Args:
        conflicts_df: 冲突记录DataFrame
        output_path: 输出文件路径
    """
    if conflicts_df.empty:
        print("没有冲突记录需要保存")
        return
    
    try:
        # 创建Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            conflicts_df.to_excel(writer, sheet_name='冲突记录', index=False)
            
            # 自动调整列宽
            worksheet = writer.sheets['冲突记录']
            for idx, col in enumerate(conflicts_df.columns):
                max_length = max(
                    conflicts_df[col].astype(str).apply(len).max(),
                    len(col)
                ) + 2
                # 限制最大宽度
                max_length = min(max_length, 80)
                worksheet.column_dimensions[chr(65 + idx)].width = max_length
        
        print(f"\n冲突记录已保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: 保存文件失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='查找Excel文件中问题相同但答案不同的冲突')
    parser.add_argument('files', nargs='+', help='要处理的Excel文件路径（支持多个文件）')
    parser.add_argument('-o', '--output', default='conflicts_output.xlsx', 
                        help='输出文件路径（默认: conflicts_output.xlsx）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("查找Excel文件中的冲突答案")
    print("=" * 60)
    
    # 加载Excel文件
    print(f"\n正在加载 {len(args.files)} 个文件...")
    data_list = load_excel_files(args.files)
    
    if not data_list:
        print("错误: 没有成功加载任何文件")
        return
    
    # 查找冲突
    print("\n正在分析冲突...")
    conflicts_df = find_conflicts(data_list)
    
    # 保存结果
    if not conflicts_df.empty:
        save_conflicts_to_excel(conflicts_df, args.output)
        
        # 显示统计信息
        print("\n" + "=" * 60)
        print("统计信息:")
        print("=" * 60)
        unique_conflicts = conflicts_df['冲突组ID'].nunique()
        print(f"发现冲突的问题数量: {unique_conflicts}")
        print(f"冲突记录总数: {len(conflicts_df)}")
        
        # 显示前5个冲突示例
        print("\n前5个冲突示例:")
        print("-" * 60)
        for conflict_id in conflicts_df['冲突组ID'].unique()[:5]:
            group = conflicts_df[conflicts_df['冲突组ID'] == conflict_id]
            print(f"\n{conflict_id} - 问题: {group.iloc[0]['问题'][:50]}...")
            for idx, row in group.iterrows():
                print(f"  答案 {idx + 1}: {row['answer'][:50]}... (来源: {row['来源文件']})")
    else:
        print("\n太好了！所有Excel文件中没有发现冲突。")


if __name__ == "__main__":
    main()

