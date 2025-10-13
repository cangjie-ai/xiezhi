# 招式二：精武判意图 - 机器学习法
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib # 用于模型保存
import time

# 1. 加载数据
data = pd.read_csv('data/intent_data_label.csv')

# 2. 中文分词函数
def chinese_tokenizer(text):
    return " ".join(jieba.cut(text))

# 3. 数据预处理
data['text_tokenized'] = data['text'].apply(chinese_tokenizer)
X = data['text_tokenized']
y = data['label']

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. 构建并训练模型管道 (Pipeline)
# Pipeline 是个好东西，能将多个步骤串联起来，非常符合架构师的思维
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(solver='liblinear')) 
])

print("开始训练模型...")
model_pipeline.fit(X_train, y_train)
print("模型训练完成!")

# 6. 评估模型
accuracy = model_pipeline.score(X_test, y_test)
print(f"模型在测试集上的准确率: {accuracy:.2f}")

# 7. 保存模型
joblib.dump(model_pipeline, 'intent_classifier_lr.pkl')
print("模型已保存到 intent_classifier_lr.pkl")

# --- 加载模型并进行预测 ---
loaded_model = joblib.load('intent_classifier_lr.pkl')

query = "被保险人去世后多久能赔？" # 规则法无法判断的句子
tokenized_query = chinese_tokenizer(query)

start_time = time.perf_counter()
prediction = loaded_model.predict([tokenized_query])[0]
proba = loaded_model.predict_proba([tokenized_query])[0]
end_time = time.perf_counter()

print(f"\n查询: '{query}'")
print(f"预测结果: {prediction} (1:寿险, 0:非寿险)")
print(f"预测概率: {proba}")
print(f"耗时: {(end_time - start_time) * 1000:.2f}ms")