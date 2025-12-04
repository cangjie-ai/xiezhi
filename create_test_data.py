"""
创建测试数据用于验证冲突查找工具
"""
import pandas as pd
from pathlib import Path


def create_test_files():
    """创建5个测试Excel文件，包含一些冲突的记录"""
    
    # 创建测试数据目录
    test_dir = Path("test_conflict_data")
    test_dir.mkdir(exist_ok=True)
    
    # 文件1
    data1 = pd.DataFrame({
        '问题': [
            '什么是人工智能？',
            '机器学习是什么？',
            'Python的优点有哪些？',
            '什么是深度学习？',
        ],
        'answer': [
            '人工智能是计算机科学的一个分支',
            '机器学习是AI的一个子领域',
            'Python简单易学，应用广泛',
            '深度学习使用神经网络',
        ]
    })
    data1.to_excel(test_dir / 'test_file1.xlsx', index=False)
    
    # 文件2
    data2 = pd.DataFrame({
        '问题': [
            '什么是人工智能 ？',  # 注意：这里有空格
            '机器学习是什么？',
            'Java的特点是什么？',
            '什么是神经网络？',
        ],
        'answer': [
            '人工智能是模拟人类智能的技术',  # 冲突：答案不同
            '机器学习是AI的一个子领域',  # 相同答案
            'Java是面向对象的编程语言',
            '神经网络模仿生物神经元',
        ]
    })
    data2.to_excel(test_dir / 'test_file2.xlsx', index=False)
    
    # 文件3
    data3 = pd.DataFrame({
        '问题': [
            '什么是\n人工智能？',  # 注意：这里有换行符
            'Python的优点有哪些？',
            '什么是云计算？',
            '数据库的作用是什么？',
        ],
        'answer': [
            '人工智能让机器具有智能',  # 冲突：第三个不同答案
            'Python语法简洁，库丰富',  # 冲突：答案不同
            '云计算提供按需计算资源',
            '数据库用于存储和管理数据',
        ]
    })
    data3.to_excel(test_dir / 'test_file3.xlsx', index=False)
    
    # 文件4
    data4 = pd.DataFrame({
        '问题': [
            '机器学习是什么？',
            '什么是区块链？',
            'React是什么？',
            'API是什么意思？',
        ],
        'answer': [
            '机器学习通过数据训练模型',  # 冲突：答案不同
            '区块链是分布式账本技术',
            'React是前端JavaScript框架',
            'API是应用程序接口',
        ]
    })
    data4.to_excel(test_dir / 'test_file4.xlsx', index=False)
    
    # 文件5
    data5 = pd.DataFrame({
        '问题': [
            '什么是深度学习？',
            'Docker的用途是什么？',
            'Git是什么？',
            'Python的优点有哪些？',
        ],
        'answer': [
            '深度学习使用神经网络',  # 相同答案
            'Docker用于容器化应用',
            'Git是版本控制系统',
            'Python简单易学，应用广泛',  # 相同答案
        ]
    })
    data5.to_excel(test_dir / 'test_file5.xlsx', index=False)
    
    print(f"测试文件已创建在 {test_dir} 目录下")
    print("\n预期冲突：")
    print("1. '什么是人工智能？' - 3个不同答案（来自file1, file2, file3）")
    print("2. 'Python的优点有哪些？' - 2个不同答案（来自file1/file5, file3）")
    print("3. '机器学习是什么？' - 2个不同答案（来自file1/file2, file4）")
    print("\n运行以下命令测试工具：")
    print(f"python find_conflicting_answers.py {test_dir}/*.xlsx -o test_conflicts.xlsx")


if __name__ == "__main__":
    create_test_files()

