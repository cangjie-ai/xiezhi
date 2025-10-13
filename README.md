# 解豸 (Xiezhi) - 意图识别模型

<div align="center">

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**解豸** - 中国古代传说中象征着公正司法的独角神兽，它能明辨是非善恶。

本项目使用解豸之名，旨在构建一个能够精准识别用户意图的智能分类系统。

</div>

---

## 📚 项目简介

本项目提供**两种意图识别方案**，用于识别用户咨询是否与寿险相关：

1. **传统机器学习**：TF-IDF + Logistic Regression（轻量快速）
2. **深度学习**：BERT 微调 + ONNX 优化（高精度）

### 应用场景
- 保险客服系统智能分流
- 用户意图预判
- 自动化业务路由

---

## 🚀 快速开始

### 环境准备

```bash
# 创建 Conda 环境
conda env create -f environment.yml

# 激活环境
conda activate xiezhi
```

### 训练模型

#### 方案 1：传统机器学习模型
```bash
python xiezhi-ml.py
```
**输出**：`intent_classifier_lr.pkl`（约 4KB，推理 <5ms）

#### 方案 2：BERT 深度学习模型
```bash
# 1. 微调 BERT 模型
python xiezhi_bert_x.py

# 2. 导出为 ONNX 格式（用于生产部署）
build_xiezhi_bert.bat
```
**输出**：
- `best_intent_model/`（PyTorch 格式，~156MB）
- `onnx_model/`（ONNX 格式，~156MB，推理 50-200ms CPU）

---

## 📊 性能对比

| 模型 | 大小 | 训练时间 | CPU 推理 | GPU 推理 | 准确率 |
|------|------|---------|---------|---------|--------|
| **传统 ML** | 4KB | ~1秒 | <5ms | N/A | ~85% |
| **BERT** | 156MB | ~2分钟 | 50-200ms | 10-30ms | ~95%+ |

---

## 📁 项目结构

```
xiezhi/
├── data/
│   └── intent_data_label.csv      # 训练数据（57 样本）
├── xiezhi-ml.py                   # 传统 ML 训练脚本
├── xiezhi_bert_x.py               # BERT 微调脚本
├── build_xiezhi_bert.bat          # ONNX 导出脚本
├── environment.yml                # Conda 环境配置
├── .gitignore                     # Git 忽略文件
├── MODELS.md                      # 模型管理说明
└── README.md                      # 本文件

# 生成的文件（不提交到 Git）
├── intent_classifier_lr.pkl       # 传统 ML 模型
├── best_intent_model/             # BERT 最佳模型
├── onnx_model/                    # ONNX 导出模型
└── results/                       # 训练检查点
```

---

## 💡 使用示例

### Python API

```python
# 使用传统 ML 模型
import joblib

model = joblib.load('intent_classifier_lr.pkl')
prediction = model.predict(["我想买份终身寿险"])
# 输出: [1]  (1=寿险意图, 0=非寿险意图)

# 使用 ONNX 模型
import onnxruntime as ort
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("./onnx_model")
session = ort.InferenceSession("./onnx_model/model.onnx")

text = "我想买份终身寿险"
inputs = tokenizer(text, padding="max_length", truncation=True, 
                   max_length=128, return_tensors="np")

outputs = session.run(None, {
    "input_ids": inputs["input_ids"].astype(np.int64),
    "attention_mask": inputs["attention_mask"].astype(np.int64),
    "token_type_ids": inputs["token_type_ids"].astype(np.int64)
})
# 输出: logits -> 预测概率
```

---

## 🔧 自定义训练

### 准备数据

修改 `data/intent_data_label.csv`，格式：
```csv
text,label
我想买份终身寿险,1
今天天气不错,0
```

- `label=1`：寿险相关意图
- `label=0`：非寿险意图

### 调整模型参数

编辑 `xiezhi_bert_x.py`：
```python
training_args = TrainingArguments(
    num_train_epochs=3,              # 训练轮次
    per_device_train_batch_size=8,   # 批次大小
    ...
)
```

---

## 📦 模型部署

### 集成到 FastAPI（推荐）

参考 [cangjie-backend](../cangjie-backend) 项目中的 `xiezhi_ml.py`。

### Docker 部署

```dockerfile
FROM python:3.12-slim

COPY onnx_model/ /app/models/
COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

CMD ["python", "serve.py"]
```

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 添加新数据
1. 编辑 `data/intent_data_label.csv`
2. 重新训练模型
3. 提交数据文件（CSV 可以提交）

### 提交代码
1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

---

## 📄 许可证

MIT License

---

## 📞 联系方式

如有问题，请提交 Issue。
