@echo off
REM Argilla Workspace 批量导出脚本 (Windows)
REM 
REM 使用方法:
REM   export_workspace.bat [workspace_name] [export_dir] [format]
REM
REM 示例:
REM   export_workspace.bat my_workspace ./exports csv
REM   export_workspace.bat my_workspace ./backups parquet

echo ========================================
echo Argilla Workspace 数据导出工具
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python
    pause
    exit /b 1
)

REM 获取参数
set WORKSPACE=%1
set EXPORT_DIR=%2
set FORMAT=%3

REM 如果没有提供参数，交互式询问
if "%WORKSPACE%"=="" (
    set /p WORKSPACE="请输入 Workspace 名称: "
)

if "%EXPORT_DIR%"=="" (
    set EXPORT_DIR=.\exports
    echo 使用默认导出目录: %EXPORT_DIR%
)

if "%FORMAT%"=="" (
    set /p FORMAT_CHOICE="选择导出格式 (1=CSV, 2=Parquet, 默认=CSV): "
    if "%FORMAT_CHOICE%"=="2" (
        set FORMAT=parquet
    ) else (
        set FORMAT=csv
    )
)

echo.
echo 导出配置:
echo   Workspace: %WORKSPACE%
echo   导出目录: %EXPORT_DIR%
echo   格式: %FORMAT%
echo.

REM 创建临时 Python 脚本
echo from argilla_dataset_utils import export_workspace > temp_export.py
echo success = export_workspace( >> temp_export.py
echo     workspace="%WORKSPACE%", >> temp_export.py
echo     export_base_dir="%EXPORT_DIR%", >> temp_export.py
echo     format="%FORMAT%" >> temp_export.py
echo ) >> temp_export.py
echo if success: >> temp_export.py
echo     print("\n导出完成!") >> temp_export.py
echo     exit(0) >> temp_export.py
echo else: >> temp_export.py
echo     print("\n导出失败!") >> temp_export.py
echo     exit(1) >> temp_export.py

REM 运行导出
echo 开始导出...
echo.
python temp_export.py

REM 保存返回码
set RESULT=%errorlevel%

REM 删除临时脚本
del temp_export.py

REM 根据结果显示消息
if %RESULT% equ 0 (
    echo.
    echo ========================================
    echo 导出成功!
    echo 导出位置: %EXPORT_DIR%\%WORKSPACE%
    echo ========================================
) else (
    echo.
    echo ========================================
    echo 导出失败，请检查错误信息
    echo ========================================
)

echo.
pause

