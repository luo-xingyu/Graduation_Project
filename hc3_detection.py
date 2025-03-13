import random
import numpy as np
from torch.cuda import seed_all
from peft import AutoPeftModelForSequenceClassification
from transformers import pipeline,AutoTokenizer,AutoModelForSequenceClassification,set_seed
import torch.nn.functional as F
import torch,re
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def predict_class_probabilities(text, model,tokenizer):
    # 分词
    tokenized_text = tokenizer(text, return_tensors="pt", max_length=512,truncation=True,padding="max_length").to(device)
    # 模型推理
    with torch.no_grad():
        outputs = model(**tokenized_text)
    logits = outputs.logits
    # 应用 softmax 函数获取概率分布
    probabilities = F.softmax(logits, dim=1)
    print(probabilities)
    # 获取每个类别的百分比概率
    class_probabilities = probabilities[0].tolist()

    return class_probabilities[1]
# 加载微调后的模型


def sentence_prediction(text):
    predictions = []

    # 按照标点符号分割文本为句子
    sentences = re.split('[.]', text)
    
    for sentence in sentences:
        # 如果句子为空，则跳过
        if not sentence.strip():
            continue
        
        # 对句子进行预测
        # print(sentence)
        sentence_prediction = predict_class_probabilities(sentence)
        predictions.append(sentence_prediction)
    
    # 将预测结果转换为 NumPy 数组
    predictions_array = np.array(predictions)
    # print(predictions_array)
    
    # 沿着句子轴计算平均值
    combined_prediction = np.mean(predictions_array, axis=0)
    
    return combined_prediction


if __name__ == '__main__':
    set_seed(42)
    model = AutoPeftModelForSequenceClassification.from_pretrained("./results/AI-detector",id2label={0: "Human", 1: "AI"}).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("./results/AI-detector",use_fast=True, low_cpu_mem_usage=False)
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        padding="max_length",
        truncation=True,
        max_length=512
    )
    sample_text = '''polish, causing damage to academic originality(Khalil and Er, 2023). In this paper, we create the ChatGPT-Polish dataset to support the relevant detection algorithm. Specifically, we employ ChatGPT to output a polished abstract by entering the following command: "Polish the following paragraphs in English, your answer just needs to include the polished text.", followed by the human-written abstract.#
Mix: Malicious users are likely to mix human-written abstracts with polished abstracts to evade detection algorithms. To address this problem, we create a more challenging dataset, ChatGPT-Mix, based on the polished abstracts. Specifically, we first decompose the polished abstracts and human-written abstracts according to their semantics, then.'''
    result = classifier(sample_text)
    print(result)
    print(f"预测结果：{result[0]['label']} (置信度：{result[0]['score']:.2f})")
    predict_class_probabilities(sample_text,model,tokenizer)
