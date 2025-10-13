@echo off
REM 清理 Git 仓库中的大文件和临时文件
REM 运行前请确保已提交重要更改

echo ======================================
echo 清理 Git 仓库中的不需要的文件
echo ======================================
echo.

echo [1/5] 检查 Git 状态...
git status

echo.
echo [2/5] 从 Git 中移除已追踪的大文件（保留本地文件）...

REM 移除模型文件
git rm --cached -r best_intent_model/ 2>nul
git rm --cached -r onnx_model/ 2>nul
git rm --cached -r results/ 2>nul
git rm --cached intent_classifier_lr.pkl 2>nul

REM 移除 Python 缓存
git rm --cached -r __pycache__/ 2>nul
git rm --cached -r .pytest_cache/ 2>nul

echo.
echo [3/5] 添加 .gitignore 文件...
git add .gitignore

echo.
echo [4/5] 添加 MODELS.md 文档...
git add MODELS.md

echo.
echo [5/5] 显示清理后的状态...
git status

echo.
echo ======================================
echo 清理完成！
echo ======================================
echo.
echo 后续步骤：
echo 1. 检查上面的状态输出
echo 2. 如果满意，运行: git commit -m "Add .gitignore and remove large files"
echo 3. 然后推送: git push
echo.

pause

