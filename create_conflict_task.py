"""
创建冲突审核任务

从源数据集中检测标注冲突，生成审核数据集

FIX (2025-11-25): 修复Response API访问，使用 resp.value + resp.question_name
"""

import os
from dotenv import load_dotenv
import argilla as rg
from collections import defaultdict, Counter

# 1. 环境初始化
load_dotenv()
if "ARGILLA_API_URL" not in os.environ:
    os.environ["ARGILLA_API_URL"] = "http://localhost:6900"
if "ARGILLA_API_KEY" not in os.environ:
    os.environ["ARGILLA_API_KEY"] = "owner.apikey"

client = rg.Argilla._get_default()
print(f"Connected as: {client.me.username}")

# ==========================================
# 步骤 1: 连接源数据集 "abc"
# ==========================================
SOURCE_DATASET_NAME = "abc"
print(f"\n[Step 1] 连接源数据集 '{SOURCE_DATASET_NAME}'...")

source_ds = client.datasets(name=SOURCE_DATASET_NAME)
if not source_ds:
    print(f"错误: 找不到名为 '{SOURCE_DATASET_NAME}' 的数据集。请确保数据集存在且名称正确。")
    exit(1)

print(f"成功连接数据集: {source_ds.name} (ID: {source_ds.id})")

# ==========================================
# 步骤 2: 遍历数据，检测冲突
# ==========================================
print("\n[Step 2] 正在拉取数据并检测冲突...")

conflict_records = []
total_records = 0
processed_count = 0

# 使用迭代器遍历所有记录
# 注意：Argilla 2.x 的 records 属性是一个迭代器
for record in source_ds.records:
    total_records += 1
    
    # 提取 responses
    # 在 Argilla 2.x Client 返回的 Record 对象中，responses 通常是一个列表
    # 每个 response 对象包含 user_id, status, values 等
    
    responses = record.responses
    if not responses:
        continue

    # 提取每个用户提交的有效 label
    # 假设分类问题的名称是 "label" (如果不是，请修改此处 QUESTION_NAME)
    QUESTION_NAME = "label"
    
    user_labels = {}
    
    for resp in responses:
        # 只处理已提交的 response
        if resp.status != "submitted":
            continue
            
        # Argilla 2.x: Response对象通过question_name访问，value是实际值
        try:
            val = getattr(resp, 'value', None)
            if val is not None and hasattr(resp, 'question_name') and resp.question_name == QUESTION_NAME:
                user_labels[resp.user_id] = val
        except (AttributeError, KeyError):
            continue
    
    # 至少要有两个不同的用户提交，才可能产生人际冲突
    # (如果同一个用户提交多次不同结果，通常 UI 上是覆盖，这里暂不考虑)
    if len(user_labels) < 2:
        continue
        
    # 检查是否有冲突
    unique_labels = set(user_labels.values())
    
    if len(unique_labels) > 1:
        # print(f"  [发现冲突] Record ID: {record.id} -> 观点: {unique_labels}")
        
        # 构造冲突信息
        conflict_item = {
            "original_text": record.fields["text"], # 假设原文本字段名为 "text"
            "original_record_id": str(record.id),
            "conflict_detail": user_labels, # {user_id: label_value}
            "labels_involved": list(unique_labels),
            # 保留源数据的 metadata (如果有)
            "source_metadata": record.metadata or {} 
        }
        conflict_records.append(conflict_item)

    processed_count += 1
    if processed_count % 100 == 0:
        print(f"  已处理 {processed_count} 条有响应的记录...")

print(f"\n扫描完成。")
print(f"- 总记录数: {total_records}")
print(f"- 发现冲突: {len(conflict_records)}")

# ==========================================
# 步骤 3: 生成审核数据集 (Adjudication Dataset)
# ==========================================
if not conflict_records:
    print("没有发现冲突数据，脚本结束。")
    exit(0)

TARGET_DATASET_NAME = f"{SOURCE_DATASET_NAME}_conflicts"
print(f"\n[Step 3] 生成审核数据集 '{TARGET_DATASET_NAME}'...")

target_ds = client.datasets(name=TARGET_DATASET_NAME)
if target_ds:
    print(f"删除旧的审核数据集: {TARGET_DATASET_NAME}")
    target_ds.delete()

# 定义审核数据集的 Settings
# 1. 展示原始文本
text_field = rg.TextField(name="text", title="原始内容", use_markdown=False)

# 2. 展示冲突详情 (用 Markdown 表格展示谁选了什么)
conflict_info_field = rg.TextField(
    name="conflict_info", 
    title="冲突详情 (Annotator vs Label)", 
    use_markdown=True
)

# 3. 审核员的最终裁决 (LabelQuestion)
# 动态收集所有涉及的标签，作为选项
all_involved_labels = set()
for item in conflict_records:
    all_involved_labels.update(item["labels_involved"])
    
final_decision_question = rg.LabelQuestion(
    name="final_decision",
    title="最终裁决",
    labels=list(all_involved_labels), # 使用所有冲突中出现过的标签
    description="请审核冲突并做出最终判断"
)

# 4. 审核员的备注 (TextQuestion)
reason_question = rg.TextQuestion(
    name="reason",
    title="裁决理由",
    required=False
)

settings = rg.Settings(
    fields=[text_field, conflict_info_field],
    questions=[final_decision_question, reason_question],
    allow_extra_metadata=True
)

target_ds = rg.Dataset(name=TARGET_DATASET_NAME, settings=settings)
target_ds.create()

# ==========================================
# 步骤 4: 转换并推送冲突数据
# ==========================================
print(f"\n[Step 4] 推送数据到 '{TARGET_DATASET_NAME}'...")
records_to_log = []

# 获取用户名映射 (User ID -> Username)，为了在表格里显示更友好的名字
# 注意：Admin 可能没有权限列出所有用户，这里做异常处理
user_map = {}
try:
    user_map = {u.id: u.username for u in client.users}
except Exception as e:
    print(f"警告: 无法获取用户列表 (权限不足?)，将直接显示 User ID。错误: {e}")

for item in conflict_records:
    # 构造 HTML 展示 (比 Markdown 表格更美观)
    html_content = '<div style="display: flex; flex-direction: column; gap: 8px;">'
    for uid, label in item["conflict_detail"].items():
        username = user_map.get(uid, str(uid)[:8]) # 如果找不到用户名，显示 ID 前8位
        
        # 为每个标注生成一个卡片/行
        row_html = f"""
        <div style="display: flex; align-items: center; justify-content: space-between; padding: 8px 12px; background-color: #f5f5f5; border-radius: 6px; border: 1px solid #e0e0e0;">
            <span style="font-weight: 500; color: #555;">👤 {username}</span>
            <span style="background-color: #e3f2fd; color: #1565c0; padding: 4px 10px; border-radius: 12px; font-weight: bold; font-size: 0.9em;">
                {label}
            </span>
        </div>
        """
        html_content += row_html
    html_content += '</div>'
    
    # 构造 metadata
    meta = item["source_metadata"].copy()
    meta["source_dataset"] = SOURCE_DATASET_NAME
    meta["original_record_id"] = item["original_record_id"]
    meta["conflict_type"] = "disagreement"
    
    rec = {
        "text": item["original_text"],
        "conflict_info": html_content, # 这里现在是 HTML 字符串
        "metadata": meta
    }
    records_to_log.append(rec)

target_ds.records.log(records_to_log)

print(f"\n成功推送 {len(records_to_log)} 条冲突记录。")
print(f"数据集名称: {TARGET_DATASET_NAME}")
print("请通知审核员登录 UI 进行最终裁决。")
