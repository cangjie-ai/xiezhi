"""
使用LLM API进行数据质量检查 - 检测标注错误和噪声数据

主要功能：
1. 加载标注数据
2. 通过API调用LLM模型获取预测概率
3. 使用Cleanlab检测标签错误
4. 生成质量报告和可疑样本列表
"""

import pandas as pd
import numpy as np
import requests
import json
from tqdm import tqdm
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
import warnings
import time
warnings.filterwarnings('ignore')

# ============================================
# 配置参数
# ============================================
DATA_PATH = 'data/intent_data_label.csv'  # 输入数据路径
OUTPUT_DIR = './cleanlab_results_llm'     # 输出目录
LLM_API_URL = 'http://localhost:8000/api/predict'  # LLM API地址
API_TIMEOUT = 30                           # API超时时间（秒）
BATCH_SIZE = 1                             # API批量调用大小（建议1，避免超时）
MAX_RETRIES = 3                            # 失败重试次数
RETRY_DELAY = 1                            # 重试延迟（秒）

print("=" * 70)
print("使用LLM API进行数据质量检查")
print("=" * 70)

# ============================================
# 第1步：加载和预处理数据
# ============================================
print("\n第1步：加载数据...")
df = pd.read_csv(DATA_PATH)
print(f"数据总量: {len(df)} 条")

# 数据预处理：合并system和human作为输入文本
def prepare_text(row):
    """将system和human合并为LLM输入"""
    system_col = [col for col in df.columns if 'system' in col.lower()][0]
    human_col = [col for col in df.columns if 'human' in col.lower()][0]
    
    system_text = str(row[system_col]) if pd.notna(row[system_col]) else ""
    human_text = str(row[human_col]) if pd.notna(row[human_col]) else ""
    
    if system_text:
        return f"{system_text} {human_text}"
    else:
        return human_text

df['text'] = df.apply(prepare_text, axis=1)

# 提取标签（二分类：0-拒识, 1-寿险相关）
answer_col = [col for col in df.columns if 'answer' in col.lower() or '答案' in col][0]

def extract_label(answer):
    """
    提取二分类标签：
    0: 拒识（非寿险问题）
    1: 寿险相关
    """
    answer_str = str(answer).lower()
    
    # 判断是否是寿险相关
    if '寿险' in answer_str or 'life insurance' in answer_str or '定期寿' in answer_str or '终身寿' in answer_str:
        return 1
    
    # 其他都是拒识
    return 0

df['label'] = df[answer_col].apply(extract_label)

# 保留有效数据
df_processed = df[['text', 'label']].copy()
df_processed = df_processed[df_processed['text'].str.strip() != '']
df_processed = df_processed.reset_index(drop=True)

print(f"有效数据量: {len(df_processed)} 条")
print(f"\n标签分布:")
print(f"  类别0 (拒识): {(df_processed['label']==0).sum()} 条 ({(df_processed['label']==0).sum()/len(df_processed):.1%})")
print(f"  类别1 (寿险): {(df_processed['label']==1).sum()} 条 ({(df_processed['label']==1).sum()/len(df_processed):.1%})")

# ============================================
# 第2步：定义LLM API调用函数
# ============================================
print("\n" + "=" * 70)
print("第2步：配置LLM API调用...")

def call_llm_api(text, retries=MAX_RETRIES):
    """
    调用LLM API获取预测结果
    
    API返回格式示例：
    {
        "prediction": 1,                    # 预测类别 (0 或 1)
        "probabilities": [0.2, 0.8],       # 各类别概率 [P(类别0), P(类别1)]
        "confidence": 0.8                   # 预测置信度（可选）
    }
    
    或者如果API只返回文本答案，需要解析：
    {
        "answer": "这是关于寿险的问题...",
        "intent": "寿险相关"
    }
    """
    for attempt in range(retries):
        try:
            # 构造API请求
            payload = {
                "text": text,
                # 如果需要指定返回概率，添加参数
                "return_probabilities": True,
            }
            
            # 调用API
            response = requests.post(
                LLM_API_URL,
                json=payload,
                timeout=API_TIMEOUT,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 方式1: API直接返回概率分布
                if 'probabilities' in result:
                    probs = result['probabilities']
                    # 确保是2维概率且和为1
                    if len(probs) == 2 and abs(sum(probs) - 1.0) < 0.01:
                        return np.array(probs)
                
                # 方式2: API返回预测类别和置信度
                if 'prediction' in result and 'confidence' in result:
                    pred = int(result['prediction'])
                    conf = float(result['confidence'])
                    # 将置信度转换为概率分布
                    probs = np.array([1 - conf, conf]) if pred == 1 else np.array([conf, 1 - conf])
                    return probs
                
                # 方式3: API只返回预测类别（假设均匀分布置信度）
                if 'prediction' in result:
                    pred = int(result['prediction'])
                    # 使用默认置信度0.7
                    probs = np.array([0.3, 0.7]) if pred == 1 else np.array([0.7, 0.3])
                    return probs
                
                # 方式4: API返回文本，需要解析意图
                if 'intent' in result or 'answer' in result:
                    answer = result.get('intent', result.get('answer', ''))
                    if '寿险' in answer or '定期寿' in answer or '终身寿' in answer:
                        return np.array([0.3, 0.7])  # 寿险相关
                    else:
                        return np.array([0.7, 0.3])  # 拒识
                
                # 如果无法解析，返回均匀分布
                print(f"⚠️ 无法解析API响应: {result}")
                return np.array([0.5, 0.5])
            
            else:
                print(f"⚠️ API返回错误状态码: {response.status_code}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return np.array([0.5, 0.5])
        
        except requests.Timeout:
            print(f"⚠️ API超时 (尝试 {attempt + 1}/{retries})")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            return np.array([0.5, 0.5])
        
        except Exception as e:
            print(f"⚠️ API调用失败: {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            return np.array([0.5, 0.5])
    
    return np.array([0.5, 0.5])

# 测试API连接
print(f"测试API连接: {LLM_API_URL}")
test_text = "你好，我想了解定期寿险"
try:
    test_probs = call_llm_api(test_text)
    print(f"✓ API连接成功！")
    print(f"  测试输入: {test_text}")
    print(f"  返回概率: {test_probs}")
except Exception as e:
    print(f"✗ API连接失败: {e}")
    print("请检查API地址是否正确，服务是否启动")
    exit(1)

# ============================================
# 第3步：使用LLM API获取预测概率
# ============================================
print("\n" + "=" * 70)
print("第3步：调用LLM API获取所有样本的预测概率...")
print(f"预计需要时间: ~{len(df_processed) * 0.5 / 60:.1f} 分钟（假设每次调用0.5秒）")

# 存储预测概率
pred_probs = np.zeros((len(df_processed), 2))
labels = df_processed['label'].values

# 批量调用API（带进度条）
failed_count = 0
for idx, row in tqdm(df_processed.iterrows(), total=len(df_processed), desc="API推理"):
    text = row['text']
    probs = call_llm_api(text)
    
    # 检查是否返回了有效概率
    if np.allclose(probs, [0.5, 0.5]):
        failed_count += 1
    
    pred_probs[idx] = probs
    
    # 避免API限流，稍微延迟
    if idx % 100 == 0 and idx > 0:
        time.sleep(0.1)

print(f"\n推理完成！")
if failed_count > 0:
    print(f"⚠️ 有 {failed_count} 个样本API调用失败或返回默认值")

# ============================================
# 第4步：使用Cleanlab检测标签问题
# ============================================
print("\n" + "=" * 70)
print("第4步：使用Cleanlab分析数据质量...")

# 检测标签错误
label_issues = find_label_issues(
    labels=labels,
    pred_probs=pred_probs,
    return_indices_ranked_by='self_confidence',
    filter_by='both'
)

# 计算每个样本的标签质量分数
quality_scores = get_label_quality_scores(
    labels=labels,
    pred_probs=pred_probs
)

# ============================================
# 第5步：生成质量报告
# ============================================
print("\n" + "=" * 70)
print("第5步：生成数据质量报告...")

# 添加分析结果到DataFrame
df_processed['quality_score'] = quality_scores
df_processed['is_issue'] = False
df_processed.loc[label_issues, 'is_issue'] = True
df_processed['pred_label'] = np.argmax(pred_probs, axis=1)
df_processed['pred_prob_class0'] = pred_probs[:, 0]
df_processed['pred_prob_class1'] = pred_probs[:, 1]

# 统计结果
n_issues = len(label_issues)
issue_rate = n_issues / len(df_processed) * 100

print("\n" + "=" * 70)
print("数据质量检查结果 (基于LLM API)")
print("=" * 70)
print(f"总样本数: {len(df_processed)}")
print(f"检测到问题样本: {n_issues} 条 ({issue_rate:.2f}%)")
print(f"数据质量良好样本: {len(df_processed) - n_issues} 条 ({100-issue_rate:.2f}%)")

# 按类别统计问题
print(f"\n各类别问题统计:")
for label_class in [0, 1]:
    class_name = "拒识" if label_class == 0 else "寿险相关"
    class_total = (df_processed['label'] == label_class).sum()
    class_issues = ((df_processed['label'] == label_class) & (df_processed['is_issue'])).sum()
    class_issue_rate = class_issues / class_total * 100 if class_total > 0 else 0
    print(f"  类别{label_class} ({class_name}): {class_issues}/{class_total} ({class_issue_rate:.2f}%)")

# 质量分数分布
print(f"\n质量分数统计:")
print(f"  平均质量分数: {quality_scores.mean():.4f}")
print(f"  最低质量分数: {quality_scores.min():.4f}")
print(f"  最高质量分数: {quality_scores.max():.4f}")
print(f"  质量分数中位数: {np.median(quality_scores):.4f}")

# ============================================
# 第6步：保存结果
# ============================================
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 保存完整结果
output_file = f"{OUTPUT_DIR}/data_quality_report_llm.csv"
df_processed.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n完整报告已保存: {output_file}")

# 保存可疑样本
suspicious_df = df_processed[df_processed['is_issue']].copy()
suspicious_df = suspicious_df.sort_values('quality_score')
suspicious_file = f"{OUTPUT_DIR}/suspicious_samples_llm.csv"
suspicious_df.to_csv(suspicious_file, index=False, encoding='utf-8-sig')
print(f"可疑样本列表已保存: {suspicious_file}")

# 保存质量最差的前100个样本
top_issues = df_processed.nsmallest(100, 'quality_score')
top_issues_file = f"{OUTPUT_DIR}/top_100_issues_llm.csv"
top_issues.to_csv(top_issues_file, index=False, encoding='utf-8-sig')
print(f"质量最差的100个样本已保存: {top_issues_file}")

# ============================================
# 第7步：打印典型问题样本
# ============================================
print("\n" + "=" * 70)
print("质量最差的10个样本（建议人工复核）:")
print("=" * 70)

for idx, row in df_processed.nsmallest(10, 'quality_score').iterrows():
    label_name = "拒识" if row['label'] == 0 else "寿险相关"
    pred_name = "拒识" if row['pred_label'] == 0 else "寿险相关"
    
    print(f"\n样本 #{idx}")
    print(f"  文本: {row['text'][:100]}...")
    print(f"  原始标签: {row['label']} ({label_name})")
    print(f"  预测标签: {row['pred_label']} ({pred_name})")
    print(f"  质量分数: {row['quality_score']:.4f}")
    print(f"  预测概率: [拒识: {row['pred_prob_class0']:.3f}, 寿险: {row['pred_prob_class1']:.3f}]")
    print(f"  是否可疑: {'是' if row['is_issue'] else '否'}")

print("\n" + "=" * 70)
print("数据质量检查完成！")
print("=" * 70)
print("\n建议后续操作:")
print("1. 人工复核可疑样本")
print("2. 对比BERT和LLM的检测结果，找出共同标记的问题样本（更可靠）")
print("3. 修正错误标注后重新训练模型")








