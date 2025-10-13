#!/usr/bin/env python
"""
API 参数检查工具
用法: python check_api.py
"""

def check_class_signature(class_obj, class_name):
    """检查类的 __init__ 方法签名"""
    import inspect
    
    print(f"\n{'='*70}")
    print(f"  {class_name}")
    print(f"{'='*70}\n")
    
    # 1. 参数签名
    try:
        sig = inspect.signature(class_obj.__init__)
        print("📋 参数列表:")
        print("-" * 70)
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            default = param.default
            if default == inspect.Parameter.empty:
                print(f"  • {param_name} (必需)")
            else:
                print(f"  • {param_name} = {default}")
        print()
    except Exception as e:
        print(f"❌ 无法获取签名: {e}\n")
    
    # 2. 文档字符串
    docstring = class_obj.__init__.__doc__
    if docstring:
        print("📖 文档说明:")
        print("-" * 70)
        # 只显示前 20 行
        lines = docstring.strip().split('\n')[:20]
        for line in lines:
            print(f"  {line}")
        if len(docstring.strip().split('\n')) > 20:
            print("  ...")
        print()
    
    # 3. 源代码位置
    try:
        source_file = inspect.getfile(class_obj)
        print(f"📂 源代码位置:")
        print(f"  {source_file}\n")
    except Exception as e:
        print(f"❌ 无法获取源代码位置: {e}\n")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("  API 参数检查工具")
    print("="*70)
    
    # 检查 SFTTrainer
    try:
        from trl import SFTTrainer
        check_class_signature(SFTTrainer, "SFTTrainer")
    except ImportError as e:
        print(f"\n❌ 无法导入 trl.SFTTrainer: {e}")
    
    # 检查 Trainer (transformers)
    try:
        from transformers import Trainer
        check_class_signature(Trainer, "Trainer")
    except ImportError as e:
        print(f"\n❌ 无法导入 transformers.Trainer: {e}")
    
    print("\n" + "="*70)
    print("  提示: 如需查看完整文档，在 Python 中运行:")
    print("  >>> from trl import SFTTrainer")
    print("  >>> help(SFTTrainer)")
    print("="*70 + "\n")

