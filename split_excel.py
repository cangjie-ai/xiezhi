"""
Excel文件拆分脚本
将一个大的Excel文件拆分成多个小文件，每个文件保留header
"""
import pandas as pd
import os
from pathlib import Path


def split_excel(
    input_file: str,
    output_dir: str = "split_files",
    rows_per_file: int = 1000,
    output_prefix: str = "split"
):
    """
    拆分Excel文件
    
    参数:
        input_file: 输入的Excel文件路径
        output_dir: 输出目录，默认为"split_files"
        rows_per_file: 每个文件的数据行数（不包括header），默认1000行
        output_prefix: 输出文件名前缀，默认"split"
    """
    # 读取Excel文件
    print(f"正在读取文件: {input_file}")
    df = pd.read_excel(input_file)
    
    total_rows = len(df)
    print(f"总行数（不含header）: {total_rows}")
    
    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 计算需要拆分的文件数
    num_files = (total_rows + rows_per_file - 1) // rows_per_file
    print(f"将拆分成 {num_files} 个文件，每个文件最多 {rows_per_file} 行数据")
    
    # 拆分并保存
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, total_rows)
        
        # 切片数据（包含header）
        df_chunk = df.iloc[start_idx:end_idx]
        
        # 生成输出文件名
        output_file = os.path.join(output_dir, f"{output_prefix}_{i+1}.xlsx")
        
        # 保存到Excel
        df_chunk.to_excel(output_file, index=False)
        
        actual_rows = len(df_chunk)
        print(f"已保存: {output_file} ({actual_rows} 行数据 + header)")
    
    print(f"\n拆分完成！所有文件保存在: {output_dir}")


if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "your_input_file.xlsx"  # 修改为你的输入文件名
    OUTPUT_DIR = "split_files"           # 输出目录
    ROWS_PER_FILE = 1000                 # 每个文件的数据行数（不含header）
    OUTPUT_PREFIX = "split"              # 输出文件名前缀
    
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_FILE):
        print(f"错误: 找不到文件 '{INPUT_FILE}'")
        print(f"请修改脚本中的 INPUT_FILE 变量，指定正确的Excel文件路径")
    else:
        # 执行拆分
        split_excel(
            input_file=INPUT_FILE,
            output_dir=OUTPUT_DIR,
            rows_per_file=ROWS_PER_FILE,
            output_prefix=OUTPUT_PREFIX
        )

