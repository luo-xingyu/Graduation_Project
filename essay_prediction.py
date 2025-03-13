import random
import numpy as np
from sympy.core.random import sample
from torch.cuda import seed_all
from peft import AutoPeftModelForSequenceClassification
from transformers import pipeline,AutoTokenizer,AutoModelForSequenceClassification,set_seed
import torch.nn.functional as F
import torch,re
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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

if __name__ == '__main__':
    #set_seed(42)
    model = AutoModelForSequenceClassification.from_pretrained("./results/AI-detector-distilbert-distilbert-base-uncased-finetuned-sst-2-english",id2label={0: "Human", 1: "AI"}).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("./results/AI-detector-distilbert-distilbert-base-uncased-finetuned-sst-2-english")
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        padding="max_length",
        truncation=True,
        max_length=512
    )
    text_ai = ['''Abstract
    This study investigates the correlation between digital literacy competencies and academic achievement among undergraduate students. Through a mixed-methods approach combining survey analysis (n=850) and in-depth interviews (n=30), the research reveals a statistically significant positive relationship (p<0.05) between information evaluation skills and GPA scores. The findings highlight the necessity of integrating digital literacy training into university curricula.
    1. Introduction
    1.1 Research Background
    The proliferation of online learning platforms (e.g., Coursera, edX) and AI-powered research tools (如ChatGPT) has transformed modern pedagogy. However, 62% of university educators report students' inability to discern credible digital sources (Smith et al., 2023).

    1.2 Thesis Statement
    This paper contends that systematic digital literacy cultivation can enhance academic outcomes by 18-25% through three mechanisms: critical information evaluation, ethical citation practices, and multimodal content creation.
    Literature Review
    2.1 Conceptual Framework
    Digital literacy encompasses five dimensions:

    Information navigation (Hague & Payton, 2010)
    Data validation (ALA, 2021)
    Ethical referencing (APA 7th ed.)
    Technological adaptability (UNESCO, 2022)
    Collaborative knowledge construction (Johnson, 2020)
    2.2 Research Gap
    Existing studies focus on K-12 contexts (78%) versus higher education (22%), creating an empirical void in understanding adult learners' digital challenges (Meta-analysis: Chen & Li, 2024).
    Methodology
    3.1 Research Design
    Triangulated approach:
    Component	Instrument	Sample
    Quantitative	40-item Likert scale	850 undergrads
    Qualitative	Semi-structured interview	30 participants
    Experimental	Pre-post skill assessment	Control/Test groups
    3.2 Data Collection
    Stratified sampling across 6 disciplines
    IRB-approved protocols (No. DL-2025-017)
    NVivo 14 & SPSS 28 for analysis
    Results & Discussion
    4.1 Key Findings
    68% of high-GPA students (≥3.5) demonstrated advanced source evaluation skills vs. 29% in low-GPA cohort (χ²=34.72, df=1, p<0.001)
    Qualitative data revealed three competency barriers:
    Algorithmic bias recognition (42% failure rate)
    Cross-platform verification (58% non-compliance)
    AI-assisted writing ethics (67% ambiguity)
    4.2 Theoretical Implications
    The Technology-Mediated Learning (TML) model requires revision to incorporate:
    Dynamic skill scaffolding
    Discipline-specific digital thresholds
    Generative AI governance frameworks
    Conclusion
    5.1 Summary
    This study establishes empirical evidence for digital literacy's predictive value on academic success (β=0.37, SE=0.08), particularly in STEM fields (R²=0.41).
    5.2 Limitations & Future Research
    Single-institution sampling constraint
    Longitudinal effects beyond 2-year scope
    Cross-cultural comparisons needed
    5.3 Practical Recommendations
    Mandatory digital competency certification
    Faculty development workshops on AI ethics
    Institutional repositories for verified OERs''']
    text_human = ['''Language model pretraining has led to significant performance gains but careful comparison between different approaches is challenging. Training is computationally expensive, often done on private datasets of different
    sizes, and, as we will show, hyperparameter
    choices have significant impact on the final results. We present a replication study of BERT
    pretraining (Devlin et al., 2019) that carefully
    measures the impact of many key hyperparameters and training data size. We find that BERT
    was significantly undertrained, and can match
    or exceed the performance of every model
    published after it. Our best model achieves
    state-of-the-art results on GLUE, RACE and
    SQuAD. These results highlight the importance of previously overlooked design choices,
    and raise questions about the source of recently reported improvements. We release our
    models and code''']
    result = classifier(text_human)
    print(result)
    print(f"预测结果：{result[0]['label']} (置信度：{result[0]['score']:.2f})")
    predict_class_probabilities(text_human,model,tokenizer)
