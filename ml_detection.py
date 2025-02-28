from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch
import pandas as pd
import os
from datasets import Dataset
import joblib
import numpy as np
import get_paper
import re

def read_train_test(name):
    prefix = 'hc3/'  # path to the csv data from the google drive
    train_df = pd.read_csv(os.path.join(prefix, name + '_train.csv'))
    test_df = pd.read_csv(os.path.join(prefix, name + '_test.csv'))

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    return train_dataset, test_dataset

def predict_class_probabilities(text):
    # 加载模型
    lightgbm = joblib.load('./models/lightgbm.joblib')

    # 使用之前的 tfidf_vectorizer 对新数据进行 TF-IDF 特征提取
    tfidf_vectorizer = joblib.load('./models/tfidf_vectorizer.joblib')
    X_new_tfidf = tfidf_vectorizer.transform([text])

    # 预测新数据
    prediction = lightgbm.predict_proba(X_new_tfidf)

    return prediction[0][1]

def sentence_prediction(text):
    predictions = []

    # 按照标点符号分割文本为句子
    sentences = re.split('[.]', text)
    
    for sentence in sentences:
        # 如果句子为空，则跳过
        if not sentence.strip():
            continue
        
        # 对句子进行预测
        print(sentence)
        sentence_prediction = predict_class_probabilities(sentence)
        predictions.append(sentence_prediction)
    
    # 将预测结果转换为 NumPy 数组
    predictions_array = np.array(predictions)
    print(predictions_array)
    
    # 沿着句子轴计算平均值
    combined_prediction = np.mean(predictions_array, axis=0)
    
    return combined_prediction

if __name__ == '__main__':
    # Example usage:
    paper = get_paper.Paper('paper/demo.pdf')
    _,abstract,conclusion = paper.parse_pdf()
    text = abstract

    # 获取预测结果
    
    predictions = predict_class_probabilities(text)
    print(predictions)



