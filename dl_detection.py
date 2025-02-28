from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F
import numpy as np
import get_paper
import re

def predict_class_probabilities(text, model_path="./models/roberta-large-mnli"):
    # 加载模型和分词器
    #tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    #model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    #tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large-mnli")
    #model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-large-mnli")
    #model = AutoModelForSequenceClassification.from_pretrained(model_path)
    #tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 分词
    tokenized_text = tokenizer(text, return_tensors="pt", truncation=True)
    # print(tokenized_text)

    # 模型推理
    outputs = model(**tokenized_text)
    logits = outputs.logits

    # 应用 softmax 函数获取概率分布
    probabilities = F.softmax(logits, dim=1)

    # 获取每个类别的百分比概率
    class_probabilities = probabilities[0].tolist()

    return class_probabilities[1]

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
    # 使用示例
#     text = """
#     Coffee beans, the seeds of the Coffea plant, are the foundation of one of the world's most beloved beverages. With a rich history rooted in Africa, these beans are harvested from the fruit of the coffee plant and go through a meticulous process of transformation, from the initial cherry picking to the final roast.
# The journey of coffee beans includes two primary processing methods: the dry method, which relies on natural sun-drying, and the wet method, which uses water to remove the fruit layers. Each step, from harvesting to roasting, is a dance of tradition and science that shapes the bean's eventual flavor.
# Roasting is an art that brings out the beans' character, with the intensity of heat and duration determining the coffee's final taste. The beans are then ground and brewed using a variety of methods, each revealing a different aspect of their complexity.
# Coffee's impact extends beyond the cup, with potential health benefits and a significant role in the global economy. Yet, the industry is also grappling with sustainability, aiming to protect the environment and support the livelihoods of coffee farmers.
# In essence, coffee beans are more than just a commodity; they are a testament to human ingenuity and our enduring quest for the perfect cup.
#     """
    paper = get_paper.Paper('paper/fake6.pdf')
    _,abstract,conclusion = paper.parse_pdf()
    text = abstract

    # 获取预测结果
    
    predictions = predict_class_probabilities(text)
    print(predictions)
