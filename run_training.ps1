# PowerShell 脚本 - Windows环境下运行BERT训练

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan
Write-Host "BERT寿险意图分类模型 - 训练启动脚本" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

# 检查Python环境
Write-Host "`n[1/5] 检查Python环境..." -ForegroundColor Yellow
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "错误: 未找到Python，请先安装Python 3.8+" -ForegroundColor Red
    exit 1
}

# 检查依赖包
Write-Host "`n[2/5] 检查依赖包..." -ForegroundColor Yellow
$packages = @("torch", "transformers", "datasets", "pandas", "scikit-learn", "tensorboard")
foreach ($pkg in $packages) {
    python -c "import $pkg" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "警告: 缺少依赖包 $pkg" -ForegroundColor Yellow
        $install = Read-Host "是否安装所有依赖包? (y/n)"
        if ($install -eq "y") {
            Write-Host "安装依赖包..." -ForegroundColor Yellow
            if (Test-Path "requirements.txt") {
                pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
            } else {
                pip install torch transformers datasets pandas scikit-learn tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple
            }
        }
        break
    }
}

# 检查数据文件
Write-Host "`n[3/5] 检查数据文件..." -ForegroundColor Yellow
if (Test-Path "data/intent_data_label.csv") {
    Write-Host "✓ 找到数据文件: data/intent_data_label.csv" -ForegroundColor Green
} else {
    Write-Host "警告: 未找到数据文件 data/intent_data_label.csv" -ForegroundColor Yellow
    $use_example = Read-Host "是否使用示例数据? (y/n)"
    if ($use_example -eq "y") {
        New-Item -ItemType Directory -Force -Path "data" | Out-Null
        Copy-Item "data_example.csv" "data/intent_data_label.csv"
        Write-Host "✓ 已复制示例数据" -ForegroundColor Green
    } else {
        Write-Host "错误: 请准备数据文件后再运行" -ForegroundColor Red
        exit 1
    }
}

# 创建输出目录
Write-Host "`n[4/5] 创建输出目录..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "results" | Out-Null
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
Write-Host "✓ 目录已创建" -ForegroundColor Green

# 开始训练
Write-Host "`n[5/5] 开始训练模型..." -ForegroundColor Yellow
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host "=" * 60 -ForegroundColor Cyan

$start_time = Get-Date
python xz_bert.py

if ($LASTEXITCODE -eq 0) {
    $end_time = Get-Date
    $duration = $end_time - $start_time
    
    Write-Host "`n" -NoNewline
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    Write-Host "✓ 训练完成！" -ForegroundColor Green
    Write-Host "耗时: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Green
    Write-Host "=" -NoNewline -ForegroundColor Cyan
    Write-Host "=" * 60 -ForegroundColor Cyan
    
    # 询问是否运行测试
    Write-Host "`n是否立即测试模型? (y/n): " -NoNewline -ForegroundColor Yellow
    $test = Read-Host
    if ($test -eq "y") {
        Write-Host "`n开始测试模型..." -ForegroundColor Yellow
        python xz_bert_inference.py
    }
    
    # 提示查看日志
    Write-Host "`n提示: 可以使用以下命令查看训练日志:" -ForegroundColor Cyan
    Write-Host "  tensorboard --logdir=./logs" -ForegroundColor White
    
} else {
    Write-Host "`n训练失败，请检查错误信息" -ForegroundColor Red
    exit 1
}

