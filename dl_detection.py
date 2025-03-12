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
    sample_text = "In many cultures, when a man and a woman get married, it is traditional for the wife to take the husband's last name. This is because, in the past, women were often seen as property of their husbands and taking the husband's last name was a way of showing that the woman belonged to the husband's family.Today, many people still follow this tradition because they believe it is a way to show that they are a family and to show their commitment to each other. However, it is also becoming more common for couples to choose to keep their own last names or to come up with a new last name that combines both of their names. Ultimately, the decision about whether or not to change a name after marriage is a personal one and it is up to the couple to decide what is best for them."
    result = classifier(sample_text)
    print(result)
    print(f"预测结果：{result[0]['label']} (置信度：{result[0]['score']:.2f})")
    predict_class_probabilities(sample_text,model,tokenizer)
