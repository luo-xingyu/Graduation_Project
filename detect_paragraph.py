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
    _,results = extract_paragraphs(text,0)
    score = distilroberta_detectlist(results)
    ppl = analyze_textlist(results)
    return score,ppl
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
    results = []
    last_paragraph = ""
    for page_index, paragraphs in future_results:
        if paragraphs:
            # 如果有上一次的最后一个段落，合并到当前结果的第一个段落
            if last_paragraph:
                paragraphs[0] = last_paragraph + paragraphs[0]
            else:
                # 第一页不存在last_paragraph，将其保存起来
                last_paragraph = paragraphs[-1]
            # 当前页的结果除最后一段外都添加到总结果中
            if len(paragraphs) > 1:
                results.extend(paragraphs[:-1])
                last_paragraph = paragraphs[-1]
    results.append(last_paragraph)
    score = distilroberta_detectlist(results)
    #ppl = analyze_textlist(results)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"总耗时: {elapsed_time:.2f} 秒")
    return score,0

def distilroberta_detectlist(content):
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
    score = process_pdf(path)
    print(score)
    #analyze_textlist(results)
    