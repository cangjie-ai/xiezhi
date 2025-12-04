import os
import pandas as pd
from dotenv import load_dotenv
import shutil
import json

# 1. 握手：连接到你的 Argilla 服务端

if "ARGILLA_API_URL" not in os.environ:
    os.environ["ARGILLA_API_URL"] = "http://localhost:6900"
if "ARGILLA_API_KEY" not in os.environ:
    os.environ["ARGILLA_API_KEY"] = "owner.apikey"
load_dotenv()


# 导入 argilla (环境变量设置之后)
import argilla as rg
from argilla.records._dataset_records import RecordErrorHandling

# 2. 准备更丰富的数据 (包含 ID, CoT, 版本, 标注人等)
base_data = [
    {
        "id": "msg_001",
        "text": "恭喜您获得100万！请点击链接领取。",
        "cot": "1. **分析文本**：包含“获得100万”和“点击链接”等高风险词汇。\n2. **识别模式**：典型的高额回报诱导点击。\n3. **结论**：判断为垃圾信息 (Spam)。",
        "label": "Spam",
        "annotator": "model_v2",
        "version": "v1.0",
        "confidence": 0.98
    },
    {
        "id": "msg_002",
        "text": "今晚吃什么？",
        "cot": "1. **分析文本**：日常询问饮食，无诱导性。\n2. **识别模式**：普通社交对话。\n3. **结论**：判断为正常信息 (Ham)。",
        "label": "Ham",
        "annotator": "user_A",
        "version": "v1.0",
        "confidence": 0.99
    },
    {
        "id": "msg_003",
        "text": "您的验证码是5566，请勿泄露。",
        "cot": "1. **分析文本**：包含验证码和安全提示。\n2. **识别模式**：服务通知类短信。\n3. **结论**：判断为正常信息 (Ham)。",
        "label": "Ham",
        "annotator": "model_v1",
        "version": "v1.1",
        "confidence": 0.95
    }
]

# 为了触发后端自动聚合（Terms Aggregation），增加数据量
# 模拟更多数据，确保 Metadata 筛选器能自动检测到选项
data = []
for i in range(10):  # 复制 10 份，共 30 条数据
    for item in base_data:
        new_item = item.copy()
        new_item["id"] = f"{item['id']}_{i}"  # 确保 ID 唯一
        data.append(new_item)

df = pd.DataFrame(data)

# 3. 定义高级数据集配置 (Settings)
# 3.1 Fields: 展示给标注员的内容
# 原始文本
text_field = rg.TextField(
    name="text", 
    title="原始内容", 
    use_markdown=False
)
# 思维链 (CoT): 支持 Markdown 渲染，作为辅助参考
cot_field = rg.TextField(
    name="cot", 
    title="AI 思维链 (参考答案)", 
    use_markdown=True,
    required=False
)
# 附加信息: 展示 ID, 标注人等元信息
info_field = rg.TextField(
    name="info", 
    title="附加信息", 
    use_markdown=True,
    required=False
)

# 3.2 Questions: 标注员需要回答的问题
# 标签分类
# 动态从数据中提取所有标签，避免硬编码
unique_labels = df["label"].unique().tolist()
label_question = rg.LabelQuestion(
    name="label",
    title="分类标签",
    labels=unique_labels,  # 动态使用数据中存在的标签
    description="请判断该文本是否为垃圾信息"
)
# CoT 质量评分 (RatingQuestion)
cot_quality_question = rg.RatingQuestion(
    name="cot_quality",
    title="思维链质量",
    values=[1, 2, 3, 4, 5],
    description="1分：完全错误；5分：逻辑完美",
    required=False
)
# 修正建议 (TextQuestion)
feedback_question = rg.TextQuestion(
    name="feedback",
    title="修正建议",
    description="如果思维链有误，请在此提供修正",
    required=False,
    use_markdown=True
)

# 3.3 Metadata: 高级筛选能力
# 版本号筛选
# 不指定 options，让 Argilla 后端自动聚合（只要数据量足够，会自动生效）
version_metadata = rg.TermsMetadataProperty(
    name="version",
    title="数据版本"
)
# 置信度筛选 (范围筛选)
confidence_metadata = rg.FloatMetadataProperty(
    name="confidence",
    title="模型置信度",
    min=0.0, 
    max=1.0
)

# 创建 Settings
settings = rg.Settings(
    fields=[text_field, cot_field, info_field],
    questions=[label_question, cot_quality_question, feedback_question],
    metadata=[version_metadata, confidence_metadata],
    allow_extra_metadata=True
)

# 4. 创建或获取 Dataset
dataset_name = "argilla_v2_advanced_demo"
# 获取默认客户端
client = rg.Argilla._get_default()

# 尝试通过客户端查找已存在的数据集
remote_dataset = client.datasets(name=dataset_name)

if remote_dataset:
    print(f"找到已存在的数据集: {remote_dataset.name}")
    print("检测到配置更新，正在删除旧数据集重建...")
    remote_dataset.delete()
    
print(f"创建新数据集: {dataset_name}")
# 创建新的数据集对象并创建
dataset = rg.Dataset(name=dataset_name, settings=settings)
dataset.create()

# 5. 构建记录并推送
print("正在转换数据...")
records = []
for i, row in df.iterrows():
    # 构造附加信息的 Markdown 格式
    info_md = f"""
- **ID**: `{row['id']}`
- **Annotator**: {row['annotator']}
- **Version**: {row['version']}
    """
    
    record = {
        # Fields
        "text": row["text"],
        "cot": row["cot"],
        "info": info_md,
        
        # Metadata
        "metadata": {
            "version": row["version"],
            "confidence": row["confidence"],
            "source": "script_v2"
        },
        
        # Suggestion (模型预判) - 这里我们把 CSV 里的 label 当作模型的建议
        # 使用特定 key，稍后在 mapping 中指定
        "suggested_label": row["label"],
        "suggested_score": row["confidence"],
        
        # 假设 confidence > 0.9 的我们也给一个高分建议
        "suggested_quality": 5 if row["confidence"] > 0.9 else 3
    }
    records.append(record)

print(f"正在推送 {len(records)} 条记录到 Argilla...")

# 配置 Mapping：将字典中的 key 映射到 Argilla 的 Suggestion
# 语法： "key_in_dict": "question_name.suggestion"
mapping = {
    "suggested_label": "label.suggestion",
    "suggested_score": "label.suggestion.score",
    "suggested_quality": "cot_quality.suggestion"
}

dataset.records.log(records, mapping=mapping, on_error=RecordErrorHandling.IGNORE)
print("推送完成！请刷新浏览器查看新的 Dataset。")

# 6. 版本管理与导出演示
print("\n=== 版本管理与导出演示 ===")
export_dir = "argilla_export_v2"

# 清理旧导出
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)
    
print(f"正在将数据集导出到本地目录: {export_dir}")
# to_disk 会导出 settings.json, records.json 等，方便版本控制
dataset.to_disk(export_dir)

# === 自定义步骤：修复中文转义问题 ===
# Argilla 默认导出 JSON 时会对非 ASCII 字符转义。
# 我们手动读取并重新保存，确保中文可读。
records_path = os.path.join(export_dir, "records.json")
if os.path.exists(records_path):
    print("正在处理 JSON 文件以支持中文显示...")
    with open(records_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    with open(records_path, "w", encoding="utf-8") as f:
        # ensure_ascii=False 允许写入非 ASCII 字符（如中文）
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("JSON 文件处理完成！")

print(f"导出成功！您可以将 {export_dir} 目录提交到 Git 进行版本管理。")

# 演示如何从磁盘加载（注释掉，仅作示例）
# print("加载示例: loaded_dataset = rg.Dataset.from_disk('argilla_export_v2')")
