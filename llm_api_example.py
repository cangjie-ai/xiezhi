"""
LLM API适配示例 - 根据你的实际API格式修改

这个文件展示了几种常见的API格式，请根据你的实际情况修改
check_data_quality_llm.py 中的 call_llm_api 函数
"""

import requests
import json

# ============================================
# 示例1: 返回分类结果和概率（最理想）
# ============================================
def api_example_1_with_probs():
    """
    API返回格式：
    {
        "prediction": 1,                    # 预测类别
        "probabilities": [0.2, 0.8],       # [P(拒识), P(寿险)]
        "confidence": 0.8
    }
    """
    url = "http://localhost:8000/api/classify"
    payload = {
        "text": "我想了解定期寿险",
        "return_probabilities": True
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    # 直接使用返回的概率
    probs = result['probabilities']  # [0.2, 0.8]
    return probs


# ============================================
# 示例2: 只返回分类和置信度
# ============================================
def api_example_2_with_confidence():
    """
    API返回格式：
    {
        "prediction": 1,                    # 预测类别: 0-拒识, 1-寿险
        "confidence": 0.85                  # 置信度
    }
    """
    url = "http://localhost:8000/api/predict"
    payload = {"text": "我想了解定期寿险"}
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    pred = int(result['prediction'])
    conf = float(result['confidence'])
    
    # 将置信度转换为概率分布
    if pred == 1:
        probs = [1 - conf, conf]  # [0.15, 0.85]
    else:
        probs = [conf, 1 - conf]  # [0.85, 0.15]
    
    return probs


# ============================================
# 示例3: 返回文本答案（需要解析）
# ============================================
def api_example_3_text_answer():
    """
    API返回格式：
    {
        "answer": "定期寿险是一种在约定期限内提供身故保障的保险产品...",
        "intent": "寿险咨询"
    }
    """
    url = "http://localhost:8000/api/chat"
    payload = {
        "system": "你是一个保险领域的智能客服助手",
        "human": "我想了解定期寿险"
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    answer = result.get('answer', '')
    intent = result.get('intent', '')
    
    # 根据关键词判断是否寿险相关
    keywords_life_insurance = ['寿险', '定期寿', '终身寿', '身故保障', '人身保险']
    keywords_reject = ['抱歉', '只能回答', '无法', '不能']
    
    text_to_check = f"{answer} {intent}"
    
    if any(kw in text_to_check for kw in keywords_reject):
        # 拒识回答
        probs = [0.8, 0.2]
    elif any(kw in text_to_check for kw in keywords_life_insurance):
        # 寿险相关
        probs = [0.2, 0.8]
    else:
        # 不确定
        probs = [0.5, 0.5]
    
    return probs


# ============================================
# 示例4: 使用logprobs获取概率（OpenAI风格）
# ============================================
def api_example_4_openai_style():
    """
    OpenAI风格API返回格式：
    {
        "choices": [{
            "message": {
                "content": "1",  # 分类结果
                "logprobs": {
                    "content": [{
                        "token": "1",
                        "logprob": -0.5
                    }]
                }
            }
        }]
    }
    """
    import math
    
    url = "http://localhost:8000/v1/chat/completions"
    payload = {
        "model": "your-model",
        "messages": [
            {"role": "system", "content": "你是分类器。返回0表示拒识，1表示寿险相关。只返回数字。"},
            {"role": "user", "content": "我想了解定期寿险"}
        ],
        "logprobs": True,
        "top_logprobs": 2
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    # 从logprobs提取概率
    logprobs_data = result['choices'][0]['message']['logprobs']['content'][0]
    token = logprobs_data['token']
    logprob = logprobs_data['logprob']
    
    # 转换logprob为概率
    prob = math.exp(logprob)
    
    if token == "1":
        probs = [1 - prob, prob]
    else:
        probs = [prob, 1 - prob]
    
    return probs


# ============================================
# 示例5: 自定义prompt让LLM返回JSON格式
# ============================================
def api_example_5_structured_output():
    """
    使用特殊prompt让LLM返回结构化输出
    """
    url = "http://localhost:8000/api/chat"
    
    system_prompt = """你是一个意图分类器。分析用户输入，返回JSON格式：
{
    "category": 0或1,  # 0=拒识（非寿险问题）, 1=寿险相关
    "confidence": 0.0-1.0  # 置信度
}

只返回JSON，不要其他文字。"""
    
    payload = {
        "system": system_prompt,
        "human": "我想了解定期寿险"
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    # 解析LLM返回的JSON字符串
    answer = result.get('answer', '{}')
    
    try:
        parsed = json.loads(answer)
        category = int(parsed.get('category', 0))
        confidence = float(parsed.get('confidence', 0.5))
        
        if category == 1:
            probs = [1 - confidence, confidence]
        else:
            probs = [confidence, 1 - confidence]
    except:
        # 解析失败，返回默认值
        probs = [0.5, 0.5]
    
    return probs


# ============================================
# 示例6: 批量调用API（提高效率）
# ============================================
def api_example_6_batch():
    """
    如果API支持批量调用，可以显著提高速度
    """
    url = "http://localhost:8000/api/batch_predict"
    
    texts = [
        "我想了解定期寿险",
        "今天天气怎么样",
        "终身寿险的保费是多少"
    ]
    
    payload = {
        "texts": texts,
        "return_probabilities": True
    }
    
    response = requests.post(url, json=payload)
    result = response.json()
    
    # 返回格式: {"predictions": [[0.2, 0.8], [0.9, 0.1], [0.3, 0.7]]}
    all_probs = result['predictions']
    
    return all_probs


# ============================================
# 如何修改 check_data_quality_llm.py
# ============================================
"""
在 check_data_quality_llm.py 中，找到 call_llm_api 函数，
根据你的API格式修改payload和结果解析部分。

例如，如果你的API是示例2的格式：

def call_llm_api(text, retries=MAX_RETRIES):
    for attempt in range(retries):
        try:
            # 1. 修改这里：构造你的API请求格式
            payload = {"text": text}  # 根据你的API文档修改
            
            response = requests.post(
                LLM_API_URL,
                json=payload,
                timeout=API_TIMEOUT,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 2. 修改这里：解析你的API返回格式
                pred = int(result['prediction'])
                conf = float(result['confidence'])
                
                if pred == 1:
                    probs = np.array([1 - conf, conf])
                else:
                    probs = np.array([conf, 1 - conf])
                
                return probs
            
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
                continue
            return np.array([0.5, 0.5])
    
    return np.array([0.5, 0.5])
"""


# ============================================
# 测试你的API
# ============================================
if __name__ == "__main__":
    print("测试API连接...")
    
    # 选择你的API示例进行测试
    # probs = api_example_1_with_probs()
    # probs = api_example_2_with_confidence()
    # probs = api_example_3_text_answer()
    
    # 根据你的实际API测试
    url = "http://localhost:8000/api/predict"  # 修改为你的URL
    test_text = "我想了解定期寿险"
    
    try:
        payload = {"text": test_text}  # 根据你的API修改
        response = requests.post(url, json=payload, timeout=5)
        
        print(f"状态码: {response.status_code}")
        print(f"返回内容: {response.json()}")
        
        # 根据返回内容调整解析逻辑
        
    except Exception as e:
        print(f"错误: {e}")








