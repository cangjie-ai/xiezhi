"""
快速导出脚本 - 一键导出 Workspace 数据

这是一个简化的导出脚本，适合日常使用和自动化任务。

使用方法:
    python quick_export.py [workspace] [format]

示例:
    python quick_export.py my_workspace csv
    python quick_export.py my_workspace parquet
    python quick_export.py  # 交互式选择

Author: Argilla Utils
Date: 2025-11-28
"""

import sys
import os
from datetime import datetime
from argilla_dataset_utils import ArgillaDatasetManager, export_workspace


def quick_export(workspace_name=None, format="csv", export_base_dir="./exports"):
    """
    快速导出 workspace
    
    Args:
        workspace_name: Workspace 名称（None 则交互式选择）
        format: 导出格式 (csv 或 parquet)
        export_base_dir: 导出根目录
    """
    manager = ArgillaDatasetManager()
    
    # 如果没有指定 workspace，列出可用的并让用户选择
    if not workspace_name:
        print("\n" + "="*60)
        print("可用的 Workspace:")
        print("="*60)
        
        datasets = manager.list_datasets()
        workspaces = {}
        
        for ds in datasets:
            ws = ds['workspace'] or 'default'
            if ws not in workspaces:
                workspaces[ws] = []
            workspaces[ws].append(ds['name'])
        
        if not workspaces:
            print("✗ 没有找到任何 workspace")
            return False
        
        workspace_list = sorted(workspaces.keys())
        for i, ws in enumerate(workspace_list, 1):
            ds_count = len(workspaces[ws])
            print(f"  {i}. {ws} ({ds_count} 个数据集)")
        
        try:
            choice = input("\n请选择 Workspace 编号: ").strip()
            workspace_name = workspace_list[int(choice) - 1]
        except (ValueError, IndexError):
            print("✗ 无效的选择")
            return False
    
    # 显示将要导出的数据集
    print(f"\n" + "="*60)
    print(f"准备导出 Workspace: {workspace_name}")
    print("="*60)
    
    datasets = manager.list_datasets(workspace=workspace_name)
    if datasets:
        print(f"\n将导出以下 {len(datasets)} 个数据集:")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds['name']}")
    else:
        print(f"\n✗ Workspace '{workspace_name}' 中没有数据集")
        return False
    
    # 确认导出
    confirm = input(f"\n确认导出？格式: {format.upper()} (y/n): ").strip().lower()
    if confirm != 'y':
        print("取消导出")
        return False
    
    # 开始导出
    print(f"\n{'='*60}")
    print("开始导出...")
    print(f"{'='*60}\n")
    
    start_time = datetime.now()
    
    success = export_workspace(
        workspace=workspace_name,
        export_base_dir=export_base_dir,
        format=format
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # 显示结果
    print(f"\n{'='*60}")
    if success:
        print("✓ 导出完成!")
        today = datetime.now().strftime("%Y-%m-%d")
        export_path = os.path.join(export_base_dir, workspace_name, today)
        print(f"\n导出位置: {os.path.abspath(export_path)}")
        print(f"用时: {duration:.2f} 秒")
        print(f"格式: {format.upper()}")
        
        # 显示文件列表
        if os.path.exists(export_path):
            files = [f for f in os.listdir(export_path) if not f.startswith('.')]
            print(f"\n导出文件 ({len(files)} 个):")
            for f in sorted(files)[:10]:  # 只显示前 10 个
                file_path = os.path.join(export_path, f)
                size = os.path.getsize(file_path) / 1024 / 1024  # MB
                print(f"  - {f} ({size:.2f} MB)")
            if len(files) > 10:
                print(f"  ... 还有 {len(files) - 10} 个文件")
    else:
        print("✗ 导出失败")
    print(f"{'='*60}\n")
    
    return success


if __name__ == "__main__":
    # 解析命令行参数
    workspace = None
    format = "csv"
    
    if len(sys.argv) > 1:
        workspace = sys.argv[1]
    
    if len(sys.argv) > 2:
        format_arg = sys.argv[2].lower()
        if format_arg in ["csv", "parquet"]:
            format = format_arg
        else:
            print(f"⚠ 无效的格式 '{format_arg}'，使用默认格式: csv")
    
    try:
        success = quick_export(workspace_name=workspace, format=format)
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

