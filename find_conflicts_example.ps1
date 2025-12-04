# 查找Excel文件冲突的使用示例
# 在PowerShell中运行此脚本

# 方法1: 指定所有Excel文件（推荐）
python find_conflicting_answers.py `
    "data/file1.xlsx" `
    "data/file2.xlsx" `
    "data/file3.xlsx" `
    "data/file4.xlsx" `
    "data/file5.xlsx" `
    -o "冲突分析结果.xlsx"

# 方法2: 使用通配符处理某个文件夹下的所有Excel文件
# python find_conflicting_answers.py data/*.xlsx -o "冲突分析结果.xlsx"

# 方法3: 如果文件在当前目录
# python find_conflicting_answers.py file1.xlsx file2.xlsx file3.xlsx file4.xlsx file5.xlsx

# 注意：
# 1. 请将上述路径替换为你实际的Excel文件路径
# 2. -o 参数指定输出文件名（可选，默认为conflicts_output.xlsx）
# 3. Excel文件必须包含"问题"和"answer"列

