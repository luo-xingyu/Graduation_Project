from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, fitz
import numpy as np
from nltk.tokenize import sent_tokenize
from detectgpt import GPT2PPLV2 as GPT2PPL
import openai
import requests
import json
# API配置
OPENAI_API_KEY = "sk-QXFhTtcE8Eo9NpA47501AdC0Da3840Ee96DfDc693400B9Ec"
DEEPSEEK_API_KEY = "d8d0c6d8-d566-4060-8fbb-c6f517383b12"
DEEPSEEK_API_URL = "https://ark.cn-beijing.volces.com/api/v3"
def calculate_sentence_ppl(text, model, tokenizer):
    """计算整体文本和每个句子的困惑度，返回统计信息"""
    try:
        # 划分成句子
        sentences = sent_tokenize(text)
        print(f"文本已分割为 {len(sentences)} 个句子")
        
        if not sentences:
            print("警告: 文本无法正确分割为句子")
            return -1
        
        valid_sentences = []
        perplexities = []
        
        # 对每个句子计算困惑度
        for i, sentence in enumerate(sentences):
            # 忽略太短的句子
            if len(sentence.strip()) < 20:
                print(f"句子 {i+1} 太短，跳过")
                continue
                
            try:
                print(f"处理句子 {i+1}/{len(sentences)}...")
                sentence_ppl = calculate_perplexity(sentence, model, tokenizer)
                perplexities.append(sentence_ppl)
                valid_sentences.append(sentence)
                print(f"句子 {i+1} 困惑度: {sentence_ppl:.2f}")
                           
            except Exception as e:
                print(f"处理句子 {i+1} 时出错: {e}")
        
        # 确保我们有有效的结果
        if not perplexities:
            print("所有句子处理失败")
            return -1
        
        # 计算统计信息
        avg_ppl = sum(perplexities) / len(perplexities)
        max_ppl = max(perplexities)
        min_ppl = min(perplexities)
        
        print(f"句子困惑度分析: 最小={min_ppl:.2f}, 平均={avg_ppl:.2f}, 最大={max_ppl:.2f}")
        
        return avg_ppl
        
    except Exception as e:
        print(f"计算困惑度时发生错误: {e}")
        return -1  # 返回默认值
    
def calculate_perplexity_api(text, api_model="gpt-4o"):
    """使用API计算文本的困惑度
    参数:
        text: 要分析的文本
        api_model: 选择使用的API模型，'gpt4o'或'deepseek'
    """
    try:
        if len(text.strip()) < 100:
            print("文本太短，跳过")
            return -1
            
        if api_model == "gpt-4o":
            # 使用OpenAI GPT-4o API
            client = openai.OpenAI(api_key=OPENAI_API_KEY,base_url="https://apione.zen-x.com.cn/api/v1")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": text}],
                temperature=0,
                logprobs=True,
            )
            
            # 获取log概率
            logprobs = [token.logprob for token in response.choices[0].logprobs.content]
            
        elif api_model == "deepseek":
            # 使用DeepSeek API
            headers = {
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "ep-20250327155938-zngrc",
                "messages": [{"role": "user", "content": text}],
                "temperature": 0,
                "stream": False,
                "logprobs": True
            }
            
            response = requests.post(
                f"{DEEPSEEK_API_URL}/chat/completions",
                headers=headers,
                data=json.dumps(data)
            )
            
            if response.status_code != 200:
                print(f"API调用错误（{api_model}）: {response.text}")
                return -1
            response=response.json()    
            #print(response["choices"][0]["logprobs"]["content"])
            logprobs = [token["logprob"] for token in response["choices"][0]["logprobs"]["content"]]
        else:
            raise ValueError(f"不支持的API模型: {api_model}，请选择 'gpt4o' 或 'deepseek'")
            
        # 计算困惑度
        perplexity = np.exp(-np.mean(logprobs))
        
        return perplexity
        
    except Exception as e:
        print(f"计算困惑度时发生错误({api_model}): {e}")
        return -1

def calculate_perplexity(text,model,tokenizer,stride=512):
    
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    # max_length = model.config.n_positions
    max_length = model.config.max_position_embeddings
    if hasattr(model.config, "sliding_window"):
        max_length = min(max_length, model.config.sliding_window)

    nlls = []
    prev_end_loc = 0

    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    total_nll = torch.stack(nlls).sum()
    if end_loc == 0:
        raise ValueError("end_loc is zero, cannot compute perplexity")
    avg_nll = total_nll / end_loc
    if torch.isnan(avg_nll):
        raise ValueError("The computed value is NaN, check your nlls values")
    perplexity = torch.exp(avg_nll).item()
    return perplexity

def analyze_pdf(pdf_path):
    """分析PDF文档的每一页并计算平均困惑度"""
    # 加载模型和分词器（只加载一次）
    print("加载语言模型...")
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="sdpa",  # 使用PyTorch内置的SDPA
        torch_dtype=torch.float16,  # 降低精度以节省显存
        device_map="auto",
        offload_buffers=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("模型加载完成")
    
    print(f"开始分析PDF文件: {pdf_path}")
    pdf = fitz.open(pdf_path)
    
    all_perplexities = []
    page_processed = 0
    
    for page_num in range(len(pdf)):
        print(f"\n处理第 {page_num + 1} 页...")
        page = pdf[page_num]
        text = page.get_text()
        
        if not text.strip():
            print(f"第 {page_num + 1} 页没有文本内容，跳过")
            continue
            
        try:
            # 计算当前页的困惑度
            perplexity = calculate_sentence_ppl(text,model,tokenizer)
            all_perplexities.append(perplexity)
            page_processed += 1
            print(f"第 {page_num + 1} 页处理完成，困惑度: {perplexity:.2f}")
        except Exception as e:
            print(f"处理第 {page_num + 1} 页时出错: {e}")
            continue
    
    if not all_perplexities:
        print("没有成功处理任何页面")
        return None
        
    # 计算整个文档的统计信息
    avg_ppl = sum(all_perplexities) / len(all_perplexities)
    print("\n=== 文档整体分析结果 ===")
    print(f"总页数: {len(pdf)}")
    print(f"成功处理页数: {page_processed}")
    print(f"所有页面的平均困惑度: {avg_ppl:.2f}")
    print(f"最小页面困惑度: {min(all_perplexities):.2f}")
    print(f"最大页面困惑度: {max(all_perplexities):.2f}")
    pdf.close()
    return avg_ppl
def custom_piecewise_normalize_parameterized(score, threshold=0.3):
  """
  将输入值（预期在 -1 到 1 之间）根据阈值进行分段归一化：
  - 如果 score >= threshold，则线性映射到 [0.8, 1.0] 区间。
  - 如果 score < threshold，则线性映射到 [0, 0.2] 区间。

  Args:
    score: 输入的分数，预期在 -1 到 1 之间。
    threshold: 用于分段的阈值。函数假设 threshold 在 (-1, 1) 范围内
               以确保两个分段都有意义。

  Returns:
    归一化后的值，在 [0, 0.2] 或 [0.8, 1.0] 区间内。
    在 threshold 等于 -1 或 1 的极端情况下，行为可能简化。
  """
  # 定义输入和输出范围的边界
  input_min = -1.0
  input_max = 1.0
  output_lower_min = 0.0
  output_lower_max = 0.2
  output_upper_min = 0.8
  output_upper_max = 1.0

  # --- 处理 score >= threshold 的情况 ---
  if score >= threshold:
    # 输入范围是 [threshold, input_max] (长度 input_max - threshold)
    # 输出范围是 [output_upper_min, output_upper_max] (长度 output_upper_max - output_upper_min)

    # 计算输入范围的长度
    input_range_len = input_max - threshold
    # 计算输出范围的长度
    output_range_len = output_upper_max - output_upper_min

    # 处理边缘情况：如果 threshold 等于或大于 input_max，则输入范围长度为0或负
    # 此时所有 score >= threshold 的值都应映射到输出范围的上限
    if input_range_len <= 0:
        return output_upper_max

    # 计算 score 在输入范围内的相对位置（0 到 1 之间）
    # (score - 输入范围起点) / 输入范围长度
    relative_position = (score - threshold) / input_range_len

    # 将该相对位置应用到输出范围：输出范围起点 + 相对位置 * 输出范围长度
    normalized_value = output_upper_min + relative_position * output_range_len
    # 确保结果在 [0.8, 1.0] 区间内，防止浮点数精度问题导致略微超出
    return max(output_upper_min, min(normalized_value, output_upper_max))

  # --- 处理 score < threshold 的情况 ---
  else:
    # 输入范围是 [input_min, threshold) (长度 threshold - input_min)
    # 输出范围是 [output_lower_min, output_lower_max) (长度 output_lower_max - output_lower_min)

    # 计算输入范围的长度
    input_range_len = threshold - input_min
     # 计算输出范围的长度
    output_range_len = output_lower_max - output_lower_min

    # 处理边缘情况：如果 threshold 等于或小于 input_min，则输入范围长度为0或负
    # 此时所有 score < threshold 的值（如果存在且小于input_min）都应映射到输出范围的下限
    if input_range_len <= 0:
         return output_lower_min # 映射到0.0

    # 计算 score 在输入范围内的相对位置（0 到 1 之间）
    # (score - 输入范围起点) / 输入范围长度
    relative_position = (score - input_min) / input_range_len

    # 将该相对位置应用到输出范围：输出范围起点 + 相对位置 * 输出范围长度
    normalized_value = output_lower_min + relative_position * output_range_len
    # 确保结果在 [0, 0.2] 区间内，防止浮点数精度问题导致略微超出
    return max(output_lower_min, min(normalized_value, output_lower_max))
def analyze_textlist(text_list):
    #model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    import gc
    import torch
    from torch.cuda import empty_cache
    
    # 检查是否有足够的GPU内存，否则使用CPU
    try:
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        allocated_memory = torch.cuda.memory_allocated(0)
        free_memory = total_memory - reserved_memory
        print(f"GPU内存情况: 总内存 {total_memory/1024**3:.2f}GB, 空闲 {free_memory/1024**3:.2f}GB")
        
        # 如果空闲内存小于2GB，使用CPU
        device = "cuda" if free_memory > 2 * 1024**3 else "cpu"
    except:
        device = "cpu"
        print("无法获取GPU内存信息或无GPU，使用CPU模式")
        
    print(f"使用设备: {device}")
    model = GPT2PPL(device=device)
    perplexities = []     # 存储每段的困惑度
    text_lengths = []     # 存储每段的文本长度
    page_processed = 0
    
    batch_size = 10  # 每批处理的段落数
    batches = [text_list[i:i+batch_size] for i in range(0, len(text_list), batch_size)]
    
    for batch_idx, batch in enumerate(batches):
        print(f"\n处理批次 {batch_idx+1}/{len(batches)}...")
        
        # 批次处理完后清理内存
        if batch_idx > 0 and batch_idx % 3 == 0:
            if device == "cuda":
                empty_cache()
            gc.collect()
            print("清理内存缓存")
            
        for num, text in enumerate(batch):
            global_idx = batch_idx * batch_size + num
            print(f"\n处理第 {global_idx + 1}/{len(text_list)}段...")
            
            length = len(text)
            if length < 300:
                chunk_value = 50
            elif length < 600:
                chunk_value = length // 6 
            elif length < 800:
                chunk_value = length // 7 
            elif length < 1000:
                chunk_value = length // 9
            elif length < 1500:
                chunk_value = length // 10
            else:
                chunk_value = 150
            if(length<100):
                continue
            if not text.strip():
                print(f"第 {global_idx + 1} 段没有文本内容，跳过")
                continue
                
            try:
                # 计算当前段的困惑度
                prob, score, _ = model(text, chunk_value, 'v1.1')
                # 主动释放不需要的变量
                del prob
                
                perplexity = score
                text_length = len(text.strip())  # 获取文本长度（去除空白字符）
                perplexities.append(perplexity)
                text_lengths.append(text_length)
                page_processed += 1
                
                print(f"第 {global_idx + 1} 段处理完成，长度: {text_length}，困惑度: {perplexity:.2f}")
                
                # 每处理一个文本段落后清理一次内存
                if device == "cuda":
                    empty_cache()
                
            except Exception as e:
                print(f"处理第 {global_idx + 1} 段时出错: {e}")
                # 发生错误时也清理内存
                if device == "cuda":
                    empty_cache()
                gc.collect()
                continue
        
        # 每批次结束后强制清理内存
        if device == "cuda":
            empty_cache()
        gc.collect()
        print(f"批次 {batch_idx+1} 处理完成，清理内存")
    
    # 释放模型
    del model
    if device == "cuda":
        empty_cache()
    gc.collect()
    
    if not perplexities:
        print("没有成功处理任何段落")
        return None
        
    # 计算加权平均困惑度
    total_weight = sum(text_lengths)
    weighted_avg_ppl = sum(p * w for p, w in zip(perplexities, text_lengths)) / total_weight
    print(f"处理前加权平均困惑度: {weighted_avg_ppl:.2f}")
    weighted_avg_ppl = custom_piecewise_normalize_parameterized(weighted_avg_ppl)
    # 计算普通平均困惑度（用于对比）
    simple_avg_ppl = sum(perplexities) / len(perplexities)
    
    print("\n=== 文档整体分析结果 ===")
    print(f"总段数: {len(text_list)}")
    print(f"成功处理段数: {page_processed}")
    print(f"加权平均困惑度: {weighted_avg_ppl:.2f}")
    print(f"普通平均困惑度: {simple_avg_ppl:.2f}")
    print(f"最小段困惑度: {min(perplexities):.2f}")
    print(f"最大段困惑度: {max(perplexities):.2f}")
    return weighted_avg_ppl
    
if __name__ == '__main__':
    path = r"./paper/Aizawa-tf-idfMeasures.pdf"
    avg_perplexity = analyze_pdf(path)
    if avg_perplexity is not None:
        print(f"\n论文整体平均困惑度: {avg_perplexity:.2f}")
        # 根据困惑度判断文本类型
        if avg_perplexity < 30:
            print("判断结果: 非常可能是AI生成的论文（低困惑度）")
        elif avg_perplexity < 85:
            print("判断结果: 可能是AI生成的论文（中等困惑度）")
        else:
            print("判断结果: 可能是人类撰写的论文（高困惑度）")