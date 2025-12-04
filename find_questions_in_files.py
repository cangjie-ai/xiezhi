"""
在多个Excel文件中查找指定问题清单中的问题位置
"""
import pandas as pd
import re
from pathlib import Path
from typing import List, Dict, Tuple
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


def load_question_list(file_path: str) -> pd.DataFrame:
    """
    加载问题清单文件
    
    Args:
        file_path: 问题清单Excel文件路径
        
    Returns:
        包含问题的DataFrame
    """
    try:
        df = pd.read_excel(file_path)
        
        # 检查是否包含'问题'列
        if '问题' not in df.columns:
            print(f"错误: {file_path} 缺少'问题'列")
            return pd.DataFrame()
        
        # 添加标准化的问题列
        df['问题_标准化'] = df['问题'].apply(normalize_text)
        
        # 过滤掉空问题
        df = df[df['问题_标准化'] != '']
        
        print(f"成功加载问题清单: {file_path}, 共 {len(df)} 个问题")
        
        return df
        
    except Exception as e:
        print(f"错误: 无法加载问题清单 {file_path}: {e}")
        return pd.DataFrame()


def load_source_files(file_paths: List[str]) -> List[Dict]:
    """
    加载源Excel文件
    
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
            if '问题' not in df.columns:
                print(f"警告: {file_path} 缺少'问题'列，跳过此文件")
                continue
            
            # 添加标准化的问题列
            df['问题_标准化'] = df['问题'].apply(normalize_text)
            
            # 添加原始行号（Excel中的行号，从2开始，因为第1行是表头）
            df['原始行号'] = range(2, len(df) + 2)
            
            # 添加来源文件名
            file_name = Path(file_path).name
            
            data_list.append({
                'path': file_path,
                'name': file_name,
                'data': df
            })
            
            print(f"成功加载源文件: {file_name}, 共 {len(df)} 行")
            
        except Exception as e:
            print(f"错误: 无法加载 {file_path}: {e}")
    
    return data_list


def find_questions_in_sources(question_list_df: pd.DataFrame, 
                               source_data_list: List[Dict]) -> pd.DataFrame:
    """
    在源文件中查找问题清单中的问题
    
    Args:
        question_list_df: 问题清单DataFrame
        source_data_list: 源文件数据列表
        
    Returns:
        匹配结果的DataFrame
    """
    results = []
    
    # 为每个要查找的问题创建一个字典
    for idx, question_row in question_list_df.iterrows():
        question_text = question_row['问题']
        question_normalized = question_row['问题_标准化']
        
        matches = []
        
        # 在每个源文件中查找
        for source_dict in source_data_list:
            source_name = source_dict['name']
            source_df = source_dict['data']
            
            # 查找匹配的行
            matched_rows = source_df[source_df['问题_标准化'] == question_normalized]
            
            if len(matched_rows) > 0:
                for _, match_row in matched_rows.iterrows():
                    match_info = {
                        '问题清单序号': idx + 1,
                        '要查找的问题': question_text,
                        '找到位置_文件名': source_name,
                        '找到位置_行号': match_row['原始行号'],
                        '找到位置_问题原文': match_row['问题'],
                    }
                    
                    # 如果源文件有answer列，也添加进来
                    if 'answer' in match_row:
                        match_info['找到位置_答案'] = match_row['answer']
                    
                    matches.append(match_info)
        
        # 如果找到了匹配
        if matches:
            results.extend(matches)
        else:
            # 如果没找到，也记录一条
            results.append({
                '问题清单序号': idx + 1,
                '要查找的问题': question_text,
                '找到位置_文件名': '未找到',
                '找到位置_行号': '',
                '找到位置_问题原文': '',
                '找到位置_答案': '',
            })
    
    results_df = pd.DataFrame(results)
    
    # 统计信息
    total_questions = len(question_list_df)
    found_questions = len(results_df[results_df['找到位置_文件名'] != '未找到']['问题清单序号'].unique())
    not_found_questions = total_questions - found_questions
    
    print(f"\n查找完成:")
    print(f"  问题清单总数: {total_questions}")
    print(f"  找到的问题数: {found_questions}")
    print(f"  未找到的问题数: {not_found_questions}")
    print(f"  总匹配记录数: {len(results_df[results_df['找到位置_文件名'] != '未找到'])}")
    
    return results_df


def save_results_to_excel(results_df: pd.DataFrame, output_path: str):
    """
    将查找结果保存到Excel文件
    
    Args:
        results_df: 结果DataFrame
        output_path: 输出文件路径
    """
    if results_df.empty:
        print("没有结果需要保存")
        return
    
    try:
        # 创建Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 写入所有结果
            results_df.to_excel(writer, sheet_name='查找结果', index=False)
            
            # 分别写入找到的和未找到的
            found_df = results_df[results_df['找到位置_文件名'] != '未找到']
            not_found_df = results_df[results_df['找到位置_文件名'] == '未找到']
            
            if not found_df.empty:
                found_df.to_excel(writer, sheet_name='已找到', index=False)
            
            if not not_found_df.empty:
                not_found_df.to_excel(writer, sheet_name='未找到', index=False)
            
            # 自动调整列宽
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                df_to_format = results_df if sheet_name == '查找结果' else (found_df if sheet_name == '已找到' else not_found_df)
                
                for idx, col in enumerate(df_to_format.columns):
                    max_length = max(
                        df_to_format[col].astype(str).apply(len).max(),
                        len(col)
                    ) + 2
                    # 限制最大宽度
                    max_length = min(max_length, 80)
                    # Excel列字母转换
                    col_letter = chr(65 + idx) if idx < 26 else chr(65 + idx // 26 - 1) + chr(65 + idx % 26)
                    worksheet.column_dimensions[col_letter].width = max_length
        
        print(f"\n结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"错误: 保存文件失败: {e}")


def generate_summary(results_df: pd.DataFrame) -> Dict:
    """
    生成统计摘要
    
    Args:
        results_df: 结果DataFrame
        
    Returns:
        统计摘要字典
    """
    total_questions = len(results_df['问题清单序号'].unique())
    found_df = results_df[results_df['找到位置_文件名'] != '未找到']
    
    summary = {
        '问题总数': total_questions,
        '找到的问题数': len(found_df['问题清单序号'].unique()),
        '未找到的问题数': total_questions - len(found_df['问题清单序号'].unique()),
        '总匹配记录数': len(found_df),
    }
    
    # 按文件统计
    if not found_df.empty:
        file_counts = found_df['找到位置_文件名'].value_counts().to_dict()
        summary['各文件匹配数'] = file_counts
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='在源Excel文件中查找问题清单中的问题位置'
    )
    parser.add_argument('question_list', help='问题清单Excel文件路径（必须包含"问题"列）')
    parser.add_argument('source_files', nargs='+', help='源Excel文件路径列表（用于查找问题）')
    parser.add_argument('-o', '--output', default='question_locations.xlsx',
                        help='输出文件路径（默认: question_locations.xlsx）')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("在Excel文件中查找问题位置")
    print("=" * 70)
    
    # 1. 加载问题清单
    print(f"\n步骤1: 加载问题清单...")
    question_list_df = load_question_list(args.question_list)
    
    if question_list_df.empty:
        print("错误: 无法加载问题清单，程序退出")
        return
    
    # 2. 加载源文件
    print(f"\n步骤2: 加载 {len(args.source_files)} 个源文件...")
    source_data_list = load_source_files(args.source_files)
    
    if not source_data_list:
        print("错误: 没有成功加载任何源文件，程序退出")
        return
    
    # 3. 查找问题
    print(f"\n步骤3: 在源文件中查找问题...")
    results_df = find_questions_in_sources(question_list_df, source_data_list)
    
    # 4. 保存结果
    print(f"\n步骤4: 保存结果...")
    save_results_to_excel(results_df, args.output)
    
    # 5. 显示摘要
    print("\n" + "=" * 70)
    print("统计摘要")
    print("=" * 70)
    summary = generate_summary(results_df)
    print(f"问题总数: {summary['问题总数']}")
    print(f"找到的问题数: {summary['找到的问题数']}")
    print(f"未找到的问题数: {summary['未找到的问题数']}")
    print(f"总匹配记录数: {summary['总匹配记录数']}")
    
    if '各文件匹配数' in summary:
        print("\n各文件匹配统计:")
        for file_name, count in summary['各文件匹配数'].items():
            print(f"  {file_name}: {count} 条")
    
    print("\n完成!")


if __name__ == "__main__":
    main()

