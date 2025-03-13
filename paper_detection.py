import torch
import torch.nn as nn
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification,AutoTokenizer
from torch.utils.data import Dataset, DataLoader
def clean_text(text, stem=True):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text
def tokenize_and_pad(sentence, tokenizer, max_length):
    encoded = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='np'
    )
    return {"input_ids": encoded['input_ids'].squeeze(),
           "attention_masks": encoded["attention_mask"].squeeze()}
def processText(text, tokenizer):
    text = clean_text(text)
    text = tokenize_and_pad(text, tokenizer, 512)
    text_tensor = torch.tensor(text["input_ids"], dtype=torch.int32)
    text_attention = torch.tensor(text["attention_masks"], dtype=torch.int32)
    return {"input_ids": text_tensor, "attention_mask": text_attention}
def detection(paper):
    tokenizer = RobertaTokenizer.from_pretrained("./results/custom_tokenizer")
    generatedText = [processText(text, tokenizer) for text in paper]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RobertaForSequenceClassification.from_pretrained('distilroberta-base', num_labels=1)
    model.load_state_dict(torch.load('./results/model_state_dict.pth'))
    model.to(device)
    model.eval()
    eval_loader = DataLoader(generatedText, batch_size=1, shuffle=False)
    with torch.no_grad():
        for data in eval_loader:
            x = data['input_ids'].to(device)
            attention = data['attention_mask'].to(device)
            pred = model(x, attention_mask=attention)
            pred = nn.Sigmoid()(pred.logits)
            return pred
if __name__ == '__main__':
    text_human = ['''The powerful ability of ChatGPT has caused
widespread concern in the academic community. Malicious users could synthesize dummy
academic content through ChatGPT, which is
extremely harmful to academic rigor and originality. The need to develop ChatGPT-written
content detection algorithms call for large-scale
datasets. In this paper, we initially investigate the possible negative impact of ChatGPT on academia, and present a large-scale
CHatGPT-writtEn AbsTract dataset (CHEAT)
to support the development of detection algorithms. In particular, the ChatGPT-written
abstract dataset contains 35,304 synthetic abstracts, with Generation, Polish, and Mix as
prominent representatives. Based on these data,
we perform a thorough analysis of the existing
text synthesis detection algorithms. We show
that ChatGPT-written abstracts are detectable,
while the detection difficulty increases with human involvement. Our dataset is available in
https://github.com/botianzhe/CHEAT''']
    text_ai = [''' 
    Abstract
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
    text_ai2 = [''' An Information-Theoretic Perspective of TF–IDF Measures
Abstract
Term Frequency–Inverse Document Frequency (TF–IDF) is a foundational technique in information retrieval and natural language processing, widely used for text representation and feature weighting. While its empirical success is well-documented, a rigorous theoretical justification for its effectiveness remains underexplored. This paper reinterprets TF–IDF through the lens of information theory, framing its components—term frequency (TF) and inverse document frequency (IDF)—as measures of local and global information content, respectively. We demonstrate that TF–IDF naturally aligns with principles of entropy, mutual information, and Kullback-Leibler divergence, offering a unified framework to explain its ability to distinguish relevant terms in document collections. By formalizing TF–IDF as an information-theoretic optimization problem, we derive its theoretical bounds and demonstrate its relationship to maximal term discrimination under entropy constraints. Our analysis bridges the gap between heuristic weighting schemes and principled information theory, providing new insights for optimizing text-based machine learning models.
1. Introduction
TF–IDF, introduced by Spärck Jones (1972), quantifies term importance by combining term frequency (TF) within a document and inverse document frequency (IDF) across a corpus. Despite its simplicity, TF–IDF remains a cornerstone of text analysis, underpinning tasks such as search engine ranking, document clustering, and classification. Traditional explanations of TF–IDF emphasize its heuristic intuition: frequent terms in a document (TF) are weighted against their rarity in the corpus (IDF). However, such explanations lack a formal connection to information theory, which rigorously quantifies concepts like redundancy, surprise, and information value.
This paper addresses this gap by reframing TF–IDF as an information-theoretic measure. We argue that TF captures local entropy (information content within a document), while IDF reflects global surprisal (unexpectedness of a term across documents). By interpreting TF–IDF as a product of these dual perspectives, we establish its role in maximizing the mutual information between terms and documents—a critical objective in retrieval systems.
2. Information-Theoretic Foundations
2.1 Entropy and Information Content
Shannon entropy H(X)=−∑p(x)logp(x) measures the uncertainty or "surprise" associated with a random variable X. For text, the entropy of a term t in document d can be derived from its probability p(t∣d), approximated by TF. Conversely, IDF relates to the self-information I(t)=−logp(t), where p(t) is the corpus-wide probability of t.
2.2 Mutual Information and Term Discrimination
Mutual information I(T;D) quantifies how much information a term T provides about a document D. We show that TF–IDF implicitly maximizes I(T;D) by balancing specificity (high TF) and discriminability (high IDF), akin to minimizing the Kullback-Leibler divergence between term distributions in relevant and non-relevant documents.
3. Deconstructing TF–IDF via Information Theory
3.1 Term Frequency as Local Entropy
Let TF(t,d)= ∑ t ′f t ′,df t,d, where f t,d is the count of term t in document d. This normalizes to p(t∣d), aligning with the entropy contribution −p(t∣d)logp(t∣d). High TF terms thus represent low local entropy (predictable/redundant) unless balanced by IDF.
3.2 Inverse Document Frequency as Global Surprisal
IDF is defined as log n tN, where N is the total documents and n t is the number containing term t. This corresponds to −logp(t), where p(t)≈ Nn t. Rare terms (low p(t)) yield high self-information, acting as "signals" in a noisy corpus.
3.3 TF–IDF as Information Product
The product TF(t,d)×IDF(t) represents the joint contribution of local and global information. We formalize this as:
TF-IDF(t,d)∝p(t∣d)⋅I(t)=−logp(t)⋅p(t∣d),
which mirrors the expected information gain when observing t in d.
4. Optimization and Theoretical Bounds
Using the Maximal Marginal Relevance (MMR) framework, we prove that TF–IDF optimizes a trade-off between:
Minimizing local entropy: Prioritizing terms that reduce uncertainty within d.
Maximizing global divergence: Selecting terms that distinguish d from the corpus.
This dual objective aligns with the information bottleneck principle, where TF–IDF acts as a compressed representation preserving maximal discriminative information.
5. Empirical Validation
We test our theoretical claims on three benchmarks (20 Newsgroups, Reuters-21578, and Wikipedia articles) by:
Measuring correlation between TF–IDF weights and information-theoretic metrics (e.g., KL divergence).
Comparing TF–IDF against pure entropy-based and mutual information weighting schemes in classification tasks.
Results show TF–IDF achieves near-optimal performance (F1-score within 2% of information-theoretic baselines) while maintaining computational efficiency, supporting its role as a practical approximation of principled measures.
6. Conclusion
By grounding TF–IDF in information theory, we provide a rigorous justification for its enduring utility. The TF component controls local redundancy, while IDF penalizes globally redundant terms, together maximizing term-specific information gain. This perspective invites future work on adaptive TF–IDF variants optimized for domain-specific entropy landscapes, potentially enhancing interpretability and performance in NLP systems.''']
    text_mix = ['''Self-training methods such as ELMo (Peters et al.,
2018), GPT (Radford et al., 2018), BERT
(Devlin et al., 2019), XLM (Lample and Conneau,
2019), and XLNet (Yang et al., 2019) have
brought significant performance gains, but it can
be challenging to determine which aspects of
the methods contribute the most. Training is
computationally expensive, limiting the amount
of tuning that can be done, and is often done with
private training data of varying sizes, limiting
our ability to measure the effects of the modeling
advances We present a replication study of BERT pretraining (Devlin et al., 2019), which includes a
careful evaluation of the effects of hyperparmeter
tuning and training set size. We find that BERT
was significantly undertrained and propose an improved recipe for training BERT models, which
we call RoBERTa, that can match or exceed the
performance of all of the post-BERT methods.
Our modifications are simple, they include: (1)
training the model longer, with bigger batches,
over more data; (2) removing the next sentence
prediction objective; (3) training on longer sequences; and (4) dynamically changing the masking pattern applied to the training data. We also
collect a large new dataset (CC-NEWS) of comparable size to other privately used datasets, to better
control for training set size effects.
To comprehensively evaluate the effectiveness of these modifications, we conducted extensive experiments across multiple benchmarks including GLUE, SQuAD, and RACE. Our results demonstrate that RoBERTa achieves state-of-the-art performance without architectural changes, outperforming BERT by 2-5% on various NLP tasks. Crucially, the removal of NSP objectives led to improved coherence in single-document tasks, while dynamic masking prevented overfitting to fixed token patterns observed in static BERT pretraining.
The extended training duration (160% longer than BERT's original setup) and enlarged batch size (8k tokens vs BERT's 256) enabled more stable gradient estimates and better utilization of parallel computation resources. When trained on our CC-NEWS corpus—a publicly available 160GB text collection—RoBERTa exhibited remarkable domain adaptability, particularly in handling contemporary language patterns and rare vocabulary.''']
    print(detection(text_human))