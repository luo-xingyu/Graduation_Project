import nltk
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
import pandas as pd
import os
import joblib

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# def plot_roc_curve(y_true, y_scores):
#     fpr, tpr, _ = roc_curve(y_true, y_scores)
#     roc_auc = auc(fpr, tpr)
#     plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
#     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver Operating Characteristic (ROC) Curve')
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.show()
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

# def plot_pr_curve(y_true, y_scores):
#     # 计算准确率和召回率
#     precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

#     # 绘制PR曲线
#     plt.plot(recall, precision, marker='.')
#     plt.xlabel('Recall')
#     plt.ylabel('Precision')
#     plt.title('Precision-Recall Curve')
#     plt.grid(True)
#     plt.show()
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
    df = pd.read_json('data/cheat.test')
    data = Dataset.from_pandas(df)
    return data

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB


def train():
    data = read_train_test()

    # 准备数据（示例数据）
    documents = data['content']
    labels = data['label']
    print(len(documents))

    X_train, X_test, y_train, y_test = train_test_split(documents, labels, test_size=0.2, random_state=42)

    # 计算 TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    # 保存 TF-IDF 向量化器
    joblib.dump(tfidf_vectorizer, './models/tfidf_vectorizer.joblib')
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # 构建分类模型
    clf_lgbm = LGBMClassifier()
    clf_lgbm.fit(X_train_tfidf, y_train)
    print(1)

    clf_lr = LogisticRegression()
    clf_lr.fit(X_train_tfidf, y_train)
    print(2)

    clf_nb = MultinomialNB()  # 使用朴素贝叶斯
    clf_nb.fit(X_train_tfidf, y_train)
    print(3)

    # 预测
    # y_pred_lgbm = clf_lgbm.predict_proba(X_test_tfidf)[:, 1]
    # y_pred_lr = clf_lr.predict_proba(X_test_tfidf)[:, 1]
    # y_pred_nb = clf_nb.predict_proba(X_test_tfidf)[:, 1]

    # y_scores_list = [y_pred_lgbm,y_pred_lr,y_pred_nb]
    # labels = ['LR','LightGBM','Naive Bayes']
    # plot_roc_curve(y_test,y_scores_list,labels)
    # plot_pr_curve(y_test,y_scores_list,labels)

    # # 构建分类模型（这里使用Lightgbm分类器）
    clf = LGBMClassifier()
    clf.fit(X_train_tfidf, y_train)

    # # 预测
    # y_pred = clf.predict_proba(X_test_tfidf)
    # y = y_pred[:,1]

    # y_pred = clf.predict(X_test_tfidf)
    # draw_matrix(y_test,y_pred)
    # 预测
    y_pred = clf_lgbm.predict(X_test_tfidf)
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    y_pred = clf_lr.predict(X_test_tfidf)
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    y_pred = clf_nb.predict(X_test_tfidf)
    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # # 评估模型
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy: {accuracy:.2f}")

    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))

    joblib.dump(clf, './models/lightgbm.joblib')

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
    # import get_paper

    # paper = get_paper.Paper('paper/demo.pdf')
    # _,abstract,conclusion = paper.parse_pdf()
    # result = predict_class_probabilities(abstract)
    # print(result)


