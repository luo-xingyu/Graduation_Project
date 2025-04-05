import json
import os
import numpy as np
import bert_detection
from bert_detection import detect

def process_json_file(json_file_path):
    # 读取JSON文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 存储每个文本段落的预测结果和权重
    predictions = []
    weights = []
    texts = []
    
    # 遍历所有文本段落，收集有效文本
    for item in data:
        if 'text' in item and item['type'] == 'text':
            if 'REFERENCES' in item['text'] and 'text_level' in item:
                break
            if 'text_level' in item:
                continue
            text = item['text']
            # 跳过太短的文本
            if len(text) < 400:
                continue
            texts.append(text)
            # 根据文本长度设置权重
            weight = len(text)
            weights.append(weight)
            #print(f"文本: {text[:50]}...")
            #print(f"权重: {weight}")
    
    # 使用bert_detection中的detection函数处理文本
    if texts:
        # 批量处理所有文本
        all_predictions = detect(texts)
        for i in range(len(all_predictions)):
            all_predictions[i] = all_predictions[i].item()
            if all_predictions[i]>0.5:
                print("###TEXT###",texts[i],"\n")
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

if __name__ == "__main__":
    json_file_path = r"./parse_paper\Fan_Test-Time_Linear_Out-of-Distribution_Detection_CVPR_2024_paper\Fan_Test-Time_Linear_Out-of-Distribution_Detection_CVPR_2024_paper.json"
    
    if os.path.exists(json_file_path):
        result = process_json_file(json_file_path)
        #print(f"\n最终AI生成概率: {result:.4f}")
    else:
        print(f"文件不存在: {json_file_path}")
    

