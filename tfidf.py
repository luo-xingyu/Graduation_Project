import nltk
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
import pandas as pd
import os
import joblib
import time  # 导入 time 模块

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_curve(y_true, y_scores_list, labels):
    plt.figure(figsize=(8, 6))
    for y_scores, label in zip(y_scores_list, labels):
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='%s (AUC = %0.2f)' % (label, roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

def plot_pr_curve(y_true, y_scores_list, labels):
    plt.figure(figsize=(8, 6))
    for y_scores, label in zip(y_scores_list, labels):
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision, lw=1, label=label)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def draw_matrix(y_true, y_pred):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 使用热图绘制混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()


def read_train_test():
    # 读取训练集和测试集
    df1 = pd.read_csv('data/final_train.csv')
    df2 = pd.read_csv('data/final_test.csv')
    # 合并数据集
    combined_df = pd.concat([df1, df2], ignore_index=True)
    # 转换为Dataset格式
    data = Dataset.from_pandas(combined_df)
    print(f"Combined dataset length: {len(data)}")
    #print("Dataset特征:", data.column_names)
    return data

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


def train():
    data = read_train_test()

    # 准备数据（示例数据）
    documents = data['text']
    labels = data['label']
    print(f"Total documents: {len(documents)}")

    X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    # 计算 TF-IDF
    print("Calculating TF-IDF...")
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    # 保存 TF-IDF 向量化器
    joblib.dump(tfidf_vectorizer, './models/tfidf_vectorizer.joblib')
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    print("TF-IDF calculation complete.")

    # --- LightGBM ---
    print("\nTraining LightGBM...")
    lgbm_start_time = time.time() # 记录开始时间
    clf_lgbm = LGBMClassifier(random_state=42)
        
    # 正式训练并计时
    lgbm_fit_start_time = time.time()
    clf_lgbm.fit(X_train_tfidf, y_train)
    lgbm_fit_end_time = time.time()
    lgbm_fit_duration = lgbm_fit_end_time - lgbm_fit_start_time
    print(f"LightGBM training complete. Time taken: {lgbm_fit_duration:.2f} seconds")
    
    # 评估 LightGBM
    y_pred_lgbm = clf_lgbm.predict(X_test_tfidf)
    accuracy_lgbm = accuracy_score(y_test, y_pred_lgbm)
    print(f"LightGBM Accuracy: {accuracy_lgbm:.2f}")
    print("LightGBM Classification Report:")
    print(classification_report(y_test, y_pred_lgbm))
    joblib.dump(clf_lgbm, './models/lightgbm.joblib')
    print("LightGBM model saved.")


    # --- Logistic Regression ---
    print("\nTraining Logistic Regression...")
    lr_start_time = time.time() # 记录开始时间
    clf_lr = LogisticRegression(random_state=42, max_iter=1000) #增加max_iter以防不收敛
    clf_lr.fit(X_train_tfidf, y_train)
    lr_end_time = time.time() # 记录结束时间
    lr_duration = lr_end_time - lr_start_time
    print(f"Logistic Regression training complete. Time taken: {lr_duration:.2f} seconds")
    # 评估 Logistic Regression
    y_pred_lr = clf_lr.predict(X_test_tfidf)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    print(f"Logistic Regression Accuracy: {accuracy_lr:.2f}")
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, y_pred_lr))
    joblib.dump(clf_lr, './models/logistic_regression.joblib') # 保存 LR 模型
    print("Logistic Regression model saved.")

    # --- Naive Bayes ---
    print("\nTraining Naive Bayes...")
    nb_start_time = time.time() # 记录开始时间
    clf_nb = MultinomialNB()
    clf_nb.fit(X_train_tfidf, y_train)
    nb_end_time = time.time() # 记录结束时间
    nb_duration = nb_end_time - nb_start_time
    print(f"Naive Bayes training complete. Time taken: {nb_duration:.2f} seconds")
     # 评估 Naive Bayes
    y_pred_nb = clf_nb.predict(X_test_tfidf)
    accuracy_nb = accuracy_score(y_test, y_pred_nb)
    print(f"Naive Bayes Accuracy: {accuracy_nb:.2f}")
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, y_pred_nb))
    joblib.dump(clf_nb, './models/naive_bayes.joblib') # 保存 NB 模型
    print("Naive Bayes model saved.")

    # --- CatBoost ---
    print("\nTraining CatBoost...")
    cat_start_time = time.time() # 记录开始时间
    clf_cat = CatBoostClassifier(verbose=0, random_state=42)

    # 正式训练并计时
    cat_fit_start_time = time.time()
    clf_cat.fit(X_train_tfidf, y_train)
    cat_fit_end_time = time.time()
    cat_fit_duration = cat_fit_end_time - cat_fit_start_time
    print(f"CatBoost training complete. Time taken: {cat_fit_duration:.2f} seconds")

    # 评估 CatBoost
    y_pred_cat = clf_cat.predict(X_test_tfidf)
    accuracy_cat = accuracy_score(y_test, y_pred_cat)
    print(f"CatBoost Accuracy: {accuracy_cat:.2f}")
    print("CatBoost Classification Report:")
    print(classification_report(y_test, y_pred_cat))

    # 保存 CatBoost 模型
    joblib.dump(clf_cat, './models/catboost.joblib')
    print("CatBoost model saved.")


    # --- XGBoost ---
    print("\nTraining XGBoost...")
    xgb_start_time = time.time() # 记录开始时间
    clf_xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', tree_method='hist')

    # 正式训练并计时
    xgb_fit_start_time = time.time()
    clf_xgb.fit(X_train_tfidf, y_train)
    xgb_fit_end_time = time.time()
    xgb_fit_duration = xgb_fit_end_time - xgb_fit_start_time
    print(f"XGBoost training complete. Time taken: {xgb_fit_duration:.2f} seconds")

    # 评估 XGBoost
    y_pred_xgb = clf_xgb.predict(X_test_tfidf)
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    print(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb))

    # 保存 XGBoost 模型
    joblib.dump(clf_xgb, './models/xgboost.joblib')
    print("XGBoost model saved.")

    return

def predict_class_probabilities(text):
    # 加载模型
    lightgbm = joblib.load('./models/lightgbm.joblib')

    # 使用之前的 tfidf_vectorizer 对新数据进行 TF-IDF 特征提取
    tfidf_vectorizer = joblib.load('./models/tfidf_vectorizer.joblib')
    X_new_tfidf = tfidf_vectorizer.transform([text])
    
    # 获取 TF-IDF 矩阵
    tfidf_matrix = X_new_tfidf.toarray()

    # 获取特征名称（词汇）
    feature_names = tfidf_vectorizer.get_feature_names_out()
    
    # 获取 TF-IDF 权重最高的前 10 个词汇
    top_indices = tfidf_matrix.argsort()[0][-10:][::-1]
    top_words = [feature_names[i] for i in top_indices]
    top_tfidf_values = [tfidf_matrix[0][i] for i in top_indices]
    
    print("======")
    print("Top 10 words with highest TF-IDF values:")
    for word, tfidf_value in zip(top_words, top_tfidf_values):
        print(f"{word}: {tfidf_value}")
    print("======")
    
    # 预测新数据
    prediction = lightgbm.predict_proba(X_new_tfidf)
    return prediction[0][1]

def extract_adjectives(text):
    # 对文本进行分词和词性标注
    tokens = nltk.word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    # 仅保留形容词
    adjectives = [word for word, pos in tagged_tokens if pos.startswith('JJ')]
    return ' '.join(adjectives)
    

if __name__ == '__main__':
    train()
    #read_train_test()
    # import get_paper

    # paper = get_paper.Paper('paper/demo.pdf')
    # _,abstract,conclusion = paper.parse_pdf()
    # result = predict_class_probabilities(abstract)
    # print(result)


