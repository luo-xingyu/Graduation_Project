from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nltk.tokenize import sent_tokenize

def calculate_perplexity_old(text, stride=512):
    model_id = "./models/gpt2"
    model = AutoModelForCausalLM.from_pretrained(
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        attn_implementation="sdpa",  # 使用PyTorch内置的SDPA
        torch_dtype=torch.float16,  # 降低精度以节省显存
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    #model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    #tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    #model = GPT2LMHeadModel.from_pretrained(model_id)
    #tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
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
    print("Total NLL:", total_nll, "end_loc:", end_loc)
    if end_loc == 0:
        raise ValueError("end_loc is zero, cannot compute perplexity")
    value = total_nll / end_loc
    if torch.isnan(value):
        raise ValueError("The computed value is NaN, check your nlls values")
    perplexity = int(torch.exp(value))

    return perplexity

def calculate_perplexity(text):
    ppl = calculate_perplexity_old(text)
    # 划分成句子
    sentences = sent_tokenize(text)

    perplexities = []

    for sentence in sentences:
        perplexity = calculate_perplexity_old(sentence)
        # if perplexity < 85:
        #     print(sentence)
        #     print(perplexity)
        #     gltr(sentence)
        #     print('\n')
        perplexities.append(perplexity)
        # print(sentence)
        # print(perplexity)
    # print(perplexities)
    
    max_ppl = max(perplexities)
    avg_ppl = sum(perplexities) / len(perplexities)
    # print(ppl, max_ppl, avg_ppl)

    return perplexities, avg_ppl, sentences

def predict_class_probabilities(text):
    ppl, result, text = calculate_perplexity(text)
    print(result)
    if result < 20:
        return ppl, result, text
    elif result < 85:
        return ppl, result, text
    else:
        return ppl, result, text

if __name__ == '__main__':
    import get_paper
    paper = get_paper.Paper('paper/fake.pdf')
    _,abstract,conclusion = paper.parse_pdf()
    text = abstract
    calculate_perplexity(text)
