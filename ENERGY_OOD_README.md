# Energy-based OOD检测方案

## 问题背景

标准BERT分类器存在**过度自信**问题：
- 输入："今天星期几"（明显OOD）
- 模型输出：P(寿险)=0.98（虚高的概率）
- 原因：Softmax强制归一化，总要选一个类别

## 解决方案：Energy-based OOD检测

**核心思想**：用Energy分数代替Softmax概率来判断OOD

```python
Energy = -log(sum(exp(logits)))

# 训练集内样本（ID）：至少有一个logit很大 → Energy低
# 训练集外样本（OOD）：所有logits都小 → Energy高
```

## 文件说明

```
xiezhi/
├── xz_bert_binary.py                    # 二分类训练脚本（不需要第三类）
├── xz_bert_calibrate_energy.py          # Energy阈值校准工具
├── xz_bert_inference_energy.py          # Energy-based推理器
└── run_energy_calibration.py            # 一键运行脚本（推荐）
```

## 使用流程

### 方案1：一键运行（推荐）

```bash
# 1. 先训练二分类模型
python xz_bert_binary.py

# 2. 一键校准阈值并测试
python run_energy_calibration.py
```

### 方案2：分步执行

#### 步骤1：训练二分类模型

```bash
python xz_bert_binary.py
```

训练完成后会生成：
- `./best_intent_model/` - 训练好的模型
- `validation_set.csv` - 验证集（用于校准阈值）

#### 步骤2：校准Energy阈值

```bash
# 方式1：只用验证集校准（默认）
python xz_bert_calibrate_energy.py

# 方式2：使用真实OOD样本校准（如果有的话）
python xz_bert_calibrate_energy.py --with-ood
```

校准完成后会生成：
- `energy_thresholds.json` - 推荐的阈值配置

#### 步骤3：使用校准后的阈值进行推理

```python
from xz_bert_inference_energy import EnergyBasedIntentClassifier
import json

# 加载校准结果
with open('energy_thresholds.json', 'r') as f:
    thresholds = json.load(f)

moderate = thresholds['recommendations']['moderate']

# 初始化分类器
classifier = EnergyBasedIntentClassifier(
    model_path="./best_intent_model",
    energy_threshold_low=moderate['threshold_low'],
    energy_threshold_high=moderate['threshold_high'],
    temperature=1.0
)

# 预测
text = "我想了解寿险"
label, confidence = classifier.predict_single(text)
print(f"预测: {label}, 置信度: {confidence}")
```

## 阈值策略

校准工具会提供三种策略：

| 策略 | threshold_low | threshold_high | 适用场景 |
|------|---------------|----------------|----------|
| 保守 | 10%分位 | 99%分位 | 避免误拒ID样本 |
| 适中 | 25%分位 | 95%分位 | 平衡（推荐） |
| 激进 | 50%分位 | 90%分位 | 严格拒绝OOD |

## 推理决策逻辑

```
输入 → BERT → logits → 计算Energy
                          ↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
   Energy > high     low < Energy < high   Energy < low
   (明显OOD)         (不确定)              (确定ID)
        ↓                 ↓                 ↓
   拒识(OOD)         可选：LLM验证       根据logits分类
                          ↓
                    寿险相关/拒识
```

## 预期效果

**测试案例：**

| 输入 | 预期 | Energy | 判定 |
|------|------|--------|------|
| "我想买寿险" | ID | -1.5 | ✓ 寿险相关 |
| "今天星期几" | OOD | 2.8 | ✓ 拒识(OOD) |
| "区块链介绍" | OOD | 3.1 | ✓ 拒识(OOD) |
| "车险理赔" | ID | 0.2 | ✓ 拒识 |

## 高级用法

### 1. 使用真实OOD样本校准

如果您收集了一批真实OOD样本（如"今天星期几"等bad cases）：

```python
# 创建 ood_samples.csv
# 包含列：text
# 每行一个OOD样本

# 运行校准
python xz_bert_calibrate_energy.py --with-ood
```

### 2. 集成LLM二次验证

```python
from xz_bert_inference_energy import EnergyBasedIntentClassifier
from openai import OpenAI  # 或其他LLM

classifier = EnergyBasedIntentClassifier(...)
llm = OpenAI(api_key="...")

def predict_with_llm_fallback(text):
    label, confidence, details = classifier.predict_single(text, return_details=True)
    
    # 对不确定样本使用LLM
    if confidence == "medium":
        prompt = f"这是否是关于寿险的问题？问题：{text}。只回答是或否。"
        response = llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        llm_answer = response.choices[0].message.content
        
        if "是" in llm_answer:
            return "寿险相关"
        else:
            return "拒识"
    
    return label
```

### 3. 批量推理

```python
# 批量预测（高效）
texts = ["问题1", "问题2", ..., "问题N"]
results = classifier.predict_batch(texts, batch_size=32)

for result in results:
    print(f"{result['text']}: {result['prediction']} (Energy={result['energy']:.2f})")
```

## 参数调优建议

### Temperature参数

```python
# T=1.0: 标准设置（推荐）
# T>1.0: 更保守，更容易判定为OOD
# T<1.0: 更激进，更少判定为OOD

classifier = EnergyBasedIntentClassifier(
    temperature=1.0  # 从1.0开始
)
```

### 阈值调整

如果发现误判：
1. **误拒ID样本太多**：调高threshold_high
2. **OOD漏检太多**：调低threshold_high
3. 重新运行校准脚本观察分布

## 持续优化

1. **收集bad cases**：每次发现误判的OOD样本，记录下来
2. **定期重新校准**：积累100+个真实OOD样本后，使用`--with-ood`重新校准
3. **监控Energy分布**：生产环境中记录所有样本的Energy，观察漂移

## 理论依据

- Paper: [Energy-based Out-of-distribution Detection (NeurIPS 2020)](https://arxiv.org/abs/2010.03759)
- Energy分数等价于"未归一化的概率密度"
- 比Softmax概率更能反映模型的真实不确定性

## 常见问题

**Q: Energy方法是否完美？**
A: 不是，但比Softmax好很多。对于极端OOD仍可能失效。建议结合多种方法（规则、LLM等）。

**Q: 需要重新训练模型吗？**
A: 推荐重新训练二分类（效果更好），但现有三分类模型也能用。

**Q: 2万条数据够吗？**
A: 够了。校准过程很快（约1-2分钟）。

**Q: 如何选择阈值策略？**
A: 先用"适中"策略，观察效果后调整。可以在验证集上试验不同策略。

## 性能指标

在验证集上评估（假设有真实OOD样本）：
- ID接受率（TPR）: 应 > 95%
- OOD拒绝率（TNR）: 应 > 80%
- 推理速度：与原BERT相同（无额外开销）









