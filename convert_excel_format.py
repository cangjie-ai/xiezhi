"""
Excel数据格式转换脚本
将包含产品信息和问题的Excel转换为训练数据格式
"""
import pandas as pd
import json
import re
from pathlib import Path


def parse_input_field(input_text: str) -> dict:
    """
    解析input字段，提取产品名称和产品代码
    
    输入格式: "产品名称：xxxx 产品代码：xxxx"
    返回: {"产品名称": "xxxx", "产品代码": "xxxx"}
    """
    result = {}
    
    # 提取产品名称
    name_match = re.search(r'产品名称[：:]\s*(.+?)\s+产品代码', input_text)
    if name_match:
        result['产品名称'] = name_match.group(1).strip()
    
    # 提取产品代码
    code_match = re.search(r'产品代码[：:]\s*(.+?)$', input_text)
    if code_match:
        result['产品代码'] = code_match.group(1).strip()
    
    return result


def parse_output_field(output_text: str) -> list:
    """
    解析output字段，提取questions列表
    
    输入格式: '1:{"questions":["xxxx","xdsd"]}'
    返回: ["xxxx", "xdsd"]
    """
    try:
        # 清理字符串：移除 \xa0（不间断空格）和其他非标准空白字符
        cleaned_text = output_text.replace('\xa0', ' ').replace('\u00a0', ' ')
        # 移除其他可能的零宽度字符
        cleaned_text = cleaned_text.replace('\u200b', '').replace('\ufeff', '')
        
        # 移除开头的 "1:" 或类似前缀，提取JSON部分
        json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            return data.get('questions', [])
    except (json.JSONDecodeError, AttributeError) as e:
        print(f"警告: 解析output字段失败: {output_text}, 错误: {e}")
    
    return []


def convert_excel_format(
    input_file: str,
    output_file: str = "converted_output.xlsx"
):
    """
    转换Excel格式
    
    参数:
        input_file: 输入的Excel文件路径（5列格式）
        output_file: 输出的Excel文件路径（3列格式）
    """
    print(f"正在读取文件: {input_file}")
    df = pd.read_excel(input_file)
    
    # 获取列名（假设第2列是input，第4列是output）
    # 列索引从0开始，所以第2列是索引1，第4列是索引3
    input_col = df.columns[1]
    output_col = df.columns[3]
    
    print(f"输入列: {input_col}, 输出列: {output_col}")
    print(f"总行数: {len(df)}")
    
    # 存储转换后的数据
    converted_data = []
    
    # 处理每一行
    for idx, row in df.iterrows():
        input_text = str(row[input_col])
        output_text = str(row[output_col])
        
        # 解析input字段
        parsed_input = parse_input_field(input_text)
        
        # 解析output字段
        questions = parse_output_field(output_text)
        
        # 添加产品名称行
        if '产品名称' in parsed_input:
            converted_data.append({
                '系统system': 'abc',
                '模型输入human': parsed_input['产品名称'],
                '模型输出assistant': 'bcd'
            })
        
        # 添加产品代码行
        if '产品代码' in parsed_input:
            converted_data.append({
                '系统system': 'abc',
                '模型输入human': parsed_input['产品代码'],
                '模型输出assistant': 'bcd'
            })
        
        # 添加questions中的每个问题
        for question in questions:
            converted_data.append({
                '系统system': 'abc',
                '模型输入human': question,
                '模型输出assistant': 'bcd'
            })
    
    # 创建新的DataFrame
    result_df = pd.DataFrame(converted_data)
    
    print(f"\n转换完成:")
    print(f"输入行数: {len(df)}")
    print(f"输出行数: {len(result_df)}")
    
    # 保存到Excel
    result_df.to_excel(output_file, index=False)
    print(f"\n已保存到: {output_file}")


def convert_questions_format(
    input_file: str,
    output_file: str = "questions_output.xlsx"
):
    """
    转换标问和相似问格式
    
    参数:
        input_file: 输入的Excel文件路径（包含"标问"和"相似问"列）
        output_file: 输出的Excel文件路径（3列格式）
    """
    print(f"正在读取文件: {input_file}")
    df = pd.read_excel(input_file)
    
    # 检查是否包含所需列
    if '标问' not in df.columns or '相似问' not in df.columns:
        print(f"错误: Excel文件必须包含'标问'和'相似问'列")
        print(f"当前列名: {list(df.columns)}")
        return
    
    print(f"总行数: {len(df)}")
    
    # 存储转换后的数据
    converted_data = []
    
    # 处理每一行
    for idx, row in df.iterrows():
        # 处理标问列
        biaow = str(row['标问']).strip()
        if biaow and biaow != 'nan' and biaow != '0':
            converted_data.append({
                '系统system': 'abc',
                '模型输入human': biaow,
                '模型输出assistant': 'bcd'
            })
        
        # 处理相似问列
        xiangsiw = str(row['相似问']).strip()
        if xiangsiw and xiangsiw != 'nan' and xiangsiw != '0':
            # 按 | 分割
            similar_questions = [q.strip() for q in xiangsiw.split('|')]
            # 添加每个相似问
            for question in similar_questions:
                if question:  # 确保不是空字符串
                    converted_data.append({
                        '系统system': 'abc',
                        '模型输入human': question,
                        '模型输出assistant': 'bcd'
                    })
    
    # 创建新的DataFrame
    result_df = pd.DataFrame(converted_data)
    
    print(f"\n转换完成:")
    print(f"输入行数: {len(df)}")
    print(f"输出行数: {len(result_df)}")
    
    # 保存到Excel
    result_df.to_excel(output_file, index=False)
    print(f"\n已保存到: {output_file}")


def convert_intent_format(
    input_file: str,
    output_file: str = "intent_output.xlsx"
):
    """
    转换意图识别格式
    
    参数:
        input_file: 输入的Excel文件路径（包含"msg"和"是否为寿险意图"列）
        output_file: 输出的Excel文件路径（3列格式）
    """
    print(f"正在读取文件: {input_file}")
    df = pd.read_excel(input_file)
    
    # 检查是否包含所需列
    if 'msg' not in df.columns or '是否为寿险意图' not in df.columns:
        print(f"错误: Excel文件必须包含'msg'和'是否为寿险意图'列")
        print(f"当前列名: {list(df.columns)}")
        return
    
    print(f"总行数: {len(df)}")
    
    # 存储转换后的数据
    converted_data = []
    
    # 处理每一行
    for idx, row in df.iterrows():
        msg = str(row['msg']).strip()
        is_intent = str(row['是否为寿险意图']).strip()
        
        # 跳过空值
        if not msg or msg == 'nan':
            continue
        
        # 判断输出值
        if is_intent in ['是', 'YES', 'yes', 'Y', 'y', '1', 'True', 'true']:
            assistant_value = '寿险意图'
        elif is_intent in ['否', 'NO', 'no', 'N', 'n', '0', 'False', 'false']:
            assistant_value = '拒识'
        else:
            # 如果值不明确，输出警告并默认为拒识
            print(f"警告: 第{idx+2}行'是否为寿险意图'列的值不明确: '{is_intent}'，默认为'拒识'")
            assistant_value = '拒识'
        
        converted_data.append({
            '系统system': 'abc',
            '模型输入human': msg,
            '模型输出assistant': assistant_value
        })
    
    # 创建新的DataFrame
    result_df = pd.DataFrame(converted_data)
    
    print(f"\n转换完成:")
    print(f"输入行数: {len(df)}")
    print(f"输出行数: {len(result_df)}")
    
    # 统计分布
    if len(result_df) > 0:
        intent_count = (result_df['模型输出assistant'] == '寿险意图').sum()
        reject_count = (result_df['模型输出assistant'] == '拒识').sum()
        print(f"寿险意图: {intent_count} 行")
        print(f"拒识: {reject_count} 行")
    
    # 保存到Excel
    result_df.to_excel(output_file, index=False)
    print(f"\n已保存到: {output_file}")


if __name__ == "__main__":
    # ========== 配置区域 ==========
    # 转换模式: "product", "questions" 或 "intent"
    # "product": 转换产品信息和问题（5列Excel，包含input和output列）
    # "questions": 转换标问和相似问（包含"标问"和"相似问"列）
    # "intent": 转换意图识别（包含"msg"和"是否为寿险意图"列）
    MODE = "intent"  # 修改为 "product", "questions" 或 "intent"
    
    # 配置参数
    INPUT_FILE = "input_data.xlsx"      # 修改为你的输入文件名
    OUTPUT_FILE = "converted_output.xlsx"  # 输出文件名
    # ==============================
    
    # 检查输入文件是否存在
    if not Path(INPUT_FILE).exists():
        print(f"错误: 找不到文件 '{INPUT_FILE}'")
        print(f"请修改脚本中的 INPUT_FILE 变量，指定正确的Excel文件路径")
    else:
        # 根据模式执行相应的转换
        if MODE == "product":
            print("=== 模式: 转换产品信息和问题 ===\n")
            convert_excel_format(
                input_file=INPUT_FILE,
                output_file=OUTPUT_FILE
            )
        elif MODE == "questions":
            print("=== 模式: 转换标问和相似问 ===\n")
            convert_questions_format(
                input_file=INPUT_FILE,
                output_file=OUTPUT_FILE
            )
        elif MODE == "intent":
            print("=== 模式: 转换意图识别 ===\n")
            convert_intent_format(
                input_file=INPUT_FILE,
                output_file=OUTPUT_FILE
            )
        else:
            print(f"错误: 未知的模式 '{MODE}'")
            print("请将 MODE 设置为 'product', 'questions' 或 'intent'")

