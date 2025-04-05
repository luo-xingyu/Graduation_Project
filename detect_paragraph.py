import requests, time, json
from bert_detect import detect
import numpy as np
import fitz
import threading
from perplexity import analyze_textlist
from concurrent.futures import ThreadPoolExecutor

#API_KEY = "d8d0c6d8-d566-4060-8fbb-c6f517383b12" # 火山
API_KEY = "sk-QXFhTtcE8Eo9NpA47501AdC0Da3840Ee96DfDc693400B9Ec" # gpt-4o
#API_URL = "https://ark.cn-beijing.volces.com/api/v3"
API_URL = "https://apione.zen-x.com.cn/api/v1"

def getinfo(content, prompt, max_retries=3):
    """
    使用 OpenAI API 提取参考文献中的信息。
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }

    data = {
        #"model": "ep-20250327154845-jjhgt",
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "你是一个可以提取文章paragraph的助手"},
            {"role": "user", "content": prompt + content}
        ],
        "max_tokens": 12000,
        "temperature": 0.1
    }

    # 重试计数器
    retry_count = 0
    wait_time = 1  # 等待时间（秒）
    
    while retry_count < max_retries:
        try:
            response = requests.post(f"{API_URL}/chat/completions", headers=headers, data=json.dumps(data))
            
            # 检查HTTP状态码
            if response.status_code != 200:
                print(f"API返回错误状态码: {response.status_code}, 响应: {response.text}")
                raise Exception(f"HTTP错误: {response.status_code}")
                
            response_json = response.json()
            info = response_json['choices'][0]['message']['content'].strip()
            return info
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                print(f"API调用出错 (尝试 {retry_count}/{max_retries}): {e}")
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                print(f"API调用失败，已重试 {max_retries} 次: {e}")
                return ""  # 所有重试都失败后返回空字符串
    
    return ""  # 以防万一的默认返回

def extract_paragraphs(content, page_index):
    """提取文章里面每一个段落，并返回页面索引和结果"""
    start_time = time.time()
    title_prompt = '''1.将文章中的paragraph语义不产生分割的情况下提取出来,paragraph即使不完整也要提取，不同的paragraph用###分隔，第一个paragraph之前无需加###，注释，表格，图片,公式不提取，References之后的文字忽略不提取，直接返回提取结果，不要加上其它任何文字\n'''
    results = getinfo(content, title_prompt)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"页面 {page_index} 处理耗时: {elapsed_time:.2f} 秒")
    if not results:
        return page_index, []
    results = results.split('###')
    print(f"页面 {page_index} 段落数量:", len(results))
    return page_index, results

def process_page(args):
    """处理单个页面的函数，用于线程池"""
    page_content, page_index = args
    page_index, paragraphs = extract_paragraphs(page_content, page_index)
    return page_index, paragraphs
def process_text(text):
    _, paragraphs = extract_paragraphs(text, 0)
    
    # 为每个段落创建信息字典
    paragraph_info = []
    for p in paragraphs:
        paragraph_info.append({
            "text": p,
            "page": 0,  # 为纯文本添加页码，统一设为0
            "is_cross_page": False  # 纯文本处理不存在跨页
        })
    
    # 创建用于预测的段落列表
    prediction_paragraphs = [info["text"] for info in paragraph_info]
    
    # 获取各个段落预测分数
    scores = detect(prediction_paragraphs)
    for i in range(len(scores)):
        scores[i] = scores[i].item()
    
    # 计算加权平均分数
    avg_score = get_avgscore(prediction_paragraphs)
    
    # 将分数添加到段落信息中
    for i in range(len(paragraph_info)):
        if i < len(scores):
            paragraph_info[i]["score"] = scores[i]
    
    # 返回段落信息和所有分数
    return paragraph_info, avg_score, 0

def process_pdf(path):
    pdf = fitz.open(path)  # pdf文档
    text_list = [page.get_text() for page in pdf]
    start_time = time.time()
    
    # 准备线程池任务
    tasks = []
    for i, page_text in enumerate(text_list):
        tasks.append((page_text, i))
        if "References" in page_text:
            # 找到References后，处理当前页，后续页面不再处理
            break
    # 使用线程池并行处理每一页
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        future_results = list(executor.map(process_page, tasks))
    
    # 按页码排序结果
    future_results.sort(key=lambda x: x[0])
    #print(future_results[-2])
    # 合并处理结果
    paragraph_info = []  # 用于存储段落信息的列表
    last_paragraph = ""
    last_page = -1
    for page_index, paragraphs in future_results:
        if paragraphs:
            # 如果有上一次的最后一个段落，合并到当前结果的第一个段落
            if last_paragraph:
                # 跨页段落处理
                cross_paragraph = last_paragraph + paragraphs[0]
                # 添加上一页的部分段落信息
                paragraph_info.append({
                    "text": last_paragraph,
                    "page": last_page,
                    "is_cross_page": True,
                    "cross_id": len(paragraph_info)  # 用于标识跨页段落的ID
                })
                # 添加当前页的部分段落信息
                paragraph_info.append({
                    "text": paragraphs[0],
                    "page": page_index,
                    "is_cross_page": True,
                    "cross_id": len(paragraph_info) - 1  # 与上半部分共享相同的ID
                })
                
                # 添加当前页的其余段落
                if len(paragraphs) > 1:
                    for p in paragraphs[1:-1]:
                        paragraph_info.append({
                            "text": p,
                            "page": page_index,
                            "is_cross_page": False
                        })
                    last_paragraph = paragraphs[-1]
                    last_page = page_index
                else:
                    last_paragraph = ""
            else:
                # 第一页处理
                for p in paragraphs[:-1]:
                    paragraph_info.append({
                        "text": p,
                        "page": page_index,
                        "is_cross_page": False
                    })
                last_paragraph = paragraphs[-1]
                last_page = page_index
    
    # 添加最后一个段落
    if last_paragraph:
        paragraph_info.append({
            "text": last_paragraph,
            "page": last_page,
            "is_cross_page": False
        })
    
    # 创建用于预测的段落列表，将跨页段落拼接起来
    prediction_paragraphs = []
    paragraph_to_prediction_index = {}  # 映射原始段落到预测段落的索引
    prediction_index = 0
    
    # 遍历段落信息，处理跨页段落和普通段落
    i = 0
    while i < len(paragraph_info):
        info = paragraph_info[i]
        
        if info.get("is_cross_page") and info.get("cross_id") is not None:
            # 找到跨页段落的另一半
            for j in range(i+1, len(paragraph_info)):
                if paragraph_info[j].get("cross_id") == info.get("cross_id"):
                    # 拼接两部分段落
                    combined_text = info["text"] + paragraph_info[j]["text"]
                    prediction_paragraphs.append(combined_text)
                    
                    # 建立映射关系
                    paragraph_to_prediction_index[i] = prediction_index
                    paragraph_to_prediction_index[j] = prediction_index
                    
                    prediction_index += 1
                    i = j + 1  # 跳过已处理的另一半
                    break
            else:
                # 找不到另一半（不应该发生）
                prediction_paragraphs.append(info["text"])
                paragraph_to_prediction_index[i] = prediction_index
                prediction_index += 1
                i += 1
        else:
            # 普通段落
            prediction_paragraphs.append(info["text"])
            paragraph_to_prediction_index[i] = prediction_index
            prediction_index += 1
            i += 1
    
    # 各个段落预测分数
    scores = detect(prediction_paragraphs)
    for i in range(len(scores)):
        scores[i] = scores[i].item()
    avg_score = get_avgscore(prediction_paragraphs)
    # 将分数添加到段落信息中
    for i in range(len(paragraph_info)):
        if i in paragraph_to_prediction_index:
            prediction_idx = paragraph_to_prediction_index[i]
            if prediction_idx < len(scores):
                paragraph_info[i]["score"] = scores[prediction_idx]
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"总耗时: {elapsed_time:.2f} 秒")
    
    # 返回每个段落的详细信息
    return paragraph_info,avg_score,0

def get_avgscore(content):
    # 存储每个文本段落的预测结果和权重
    predictions = []
    weights = []
    texts = []
    
    # 遍历所有文本段落，收集有效文本
    for item in content:
        # 跳过太短的文本
        if len(item) < 300:
            continue
        texts.append(item)
        # 根据文本长度设置权重
        weight = len(item)
        weights.append(weight)
            
    # 使用bert_detection中的detection函数处理文本
    if texts:
        # 批量处理所有文本
        all_predictions = detect(texts)
        for i in range(len(all_predictions)):
            all_predictions[i] = all_predictions[i].item()
        print(all_predictions)

        # 将预测结果转换为NumPy数组
        predictions = np.array(all_predictions)
        weights = np.array(weights)
        
        # 归一化权重
        normalized_weights = weights / np.sum(weights)
        # 计算加权平均值
        weighted_average = np.sum(predictions * normalized_weights)
        
        print(f"\n总共处理了 {len(predictions)} 段文本")
        print(f"加权平均值: {weighted_average:.4f}")
        
        return weighted_average
    else:
        print("没有找到有效的文本段落")
        return None

if __name__ == '__main__':
    path = r"paper\dynamic-fusion-cvpr-2015.pdf"
    paragraph_info = process_pdf(path)
    print(paragraph_info)
    #analyze_textlist(results)
    