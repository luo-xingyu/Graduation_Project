#!/usr/bin/env python3

"""
T5

This code a slight modification of perplexity by hugging face
https://huggingface.co/docs/transformers/perplexity

Both this code and the orignal code are published under the MIT license.

by Burhan Ul tayyab and Nicholas Chua
"""
import time
import torch
import itertools
import math
import numpy as np
import random
import re
import transformers
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import pipeline
from transformers import T5Tokenizer
from transformers import AutoTokenizer, BartForConditionalGeneration

from collections import OrderedDict

from scipy.stats import norm
from difflib import SequenceMatcher
from multiprocessing.pool import ThreadPool

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def normCdf(x):
    return norm.cdf(x)

def likelihoodRatio(x, y):
    return normCdf(x)/normCdf(y)

torch.manual_seed(0)
np.random.seed(0)

# find a better way to abstract the class
class GPT2PPLV2:
    def __init__(self, device="cuda", model_id="gpt2-medium"):
        import gc
        
        # 智能判断使用GPU还是CPU
        if device == "cuda" and torch.cuda.is_available():
            try:
                # 检查GPU内存
                total_memory = torch.cuda.get_device_properties(0).total_memory
                reserved_memory = torch.cuda.memory_reserved(0)
                free_memory = total_memory - reserved_memory
                # 如果可用内存小于2GB，则使用CPU
                if free_memory < 2 * 1024**3:
                    print("GPU内存不足，切换到CPU模式")
                    device = "cpu"
                else:
                    print(f"使用GPU，可用内存: {free_memory/1024**3:.2f}GB")
            except Exception as e:
                print(f"检查GPU内存时出错: {e}，切换到CPU模式")
                device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("CUDA不可用，切换到CPU模式")
            device = "cpu"
            
        self.device = device
        self.model_id = model_id
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

        self.max_length = self.model.config.n_positions
        self.stride = 51
        self.threshold = 0.2

        self.t5_model = transformers.AutoModelForSeq2SeqLM.from_pretrained("t5-large").to(device).half()
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-large", model_max_length=512)

    def apply_extracted_fills(self, masked_texts, extracted_fills):
        texts = []
        for idx, (text, fills) in enumerate(zip(masked_texts, extracted_fills)):
            tokens = list(re.finditer("<extra_id_\d+>", text))
            if len(fills) < len(tokens):
                continue

            offset = 0
            for fill_idx in range(len(tokens)):
                start, end = tokens[fill_idx].span()
                text = text[:start+offset] + fills[fill_idx] + text[end+offset:]
                offset = offset - (end - start) + len(fills[fill_idx])
            texts.append(text)

        return texts

    def unmasker(self, text, num_of_masks):
        num_of_masks = max(num_of_masks)
        stop_id = self.t5_tokenizer.encode(f"<extra_id_{num_of_masks}>")[0]
        tokens = self.t5_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        for key in tokens:
            tokens[key] = tokens[key].to(self.device)

        output_sequences = self.t5_model.generate(**tokens, max_length=512, do_sample=True, top_p=0.96, num_return_sequences=1, eos_token_id=stop_id)
        results = self.t5_tokenizer.batch_decode(output_sequences, skip_special_tokens=False)

        texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in results]
        pattern = re.compile("<extra_id_\d+>")
        extracted_fills = [pattern.split(x)[1:-1] for x in texts]
        extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

        perturbed_texts = self.apply_extracted_fills(text, extracted_fills)

        return perturbed_texts


    def __call__(self, *args):
        version = args[-1]
        sentence = args[0]
        if version == "v1.1":
            return self.call_1_1(sentence, args[1])
        elif version == "v1":
            return self.call_1(sentence)
        else:
            return "Model version not defined"

#################################ppp###############
#  Version 1.1 apis
###############################################

    def replaceMask(self, text, num_of_masks):
        with torch.no_grad():
            list_generated_texts = self.unmasker(text, num_of_masks)

        return list_generated_texts

    def isSame(self, text1, text2):
        return text1 == text2

    # code took reference from https://github.com/eric-mitchell/detect-gpt
    def maskRandomWord(self, text, ratio):
        span = 2
        tokens = text.split(' ')
        mask_string = '<<<mask>>>'

        n_spans = ratio//(span + 2)

        n_masks = 0
        while n_masks < n_spans:
            start = np.random.randint(0, len(tokens) - span)
            end = start + span
            search_start = max(0, start - 1)
            search_end = min(len(tokens), end + 1)
            if mask_string not in tokens[search_start:search_end]:
                tokens[start:end] = [mask_string]
                n_masks += 1

        # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
        num_filled = 0
        for idx, token in enumerate(tokens):
            if token == mask_string:
                tokens[idx] = f'<extra_id_{num_filled}>'
                num_filled += 1
        assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
        text = ' '.join(tokens)
        return text, n_masks

    def multiMaskRandomWord(self, text, ratio, n):
        mask_texts = []
        list_num_of_masks = []
        for i in range(n):
            mask_text, num_of_masks = self.maskRandomWord(text, ratio)
            mask_texts.append(mask_text)
            list_num_of_masks.append(num_of_masks)
        return mask_texts, list_num_of_masks

    def getGeneratedTexts(self, args):
        original_text = args[0]
        n = args[1]
        texts = list(re.finditer("[^\d\W]+", original_text))
        ratio = int(0.3 * len(texts))

        mask_texts, list_num_of_masks = self.multiMaskRandomWord(original_text, ratio, n)
        list_generated_sentences = self.replaceMask(mask_texts, list_num_of_masks)
        return list_generated_sentences

    def mask(self, original_text, text, n=2, remaining=100):
        """
        text: string representing the sentence
        n: top n mask-filling to be choosen
        remaining: The remaining slots to be fill
        """

        if remaining <= 0:
            return []

        torch.manual_seed(0)
        np.random.seed(0)
        start_time = time.time()
        out_sentences = []
        pool = ThreadPool(remaining//n)
        out_sentences = pool.map(self.getGeneratedTexts, [(original_text, n) for _ in range(remaining//n)])
        out_sentences = list(itertools.chain.from_iterable(out_sentences))
        end_time = time.time()

        return out_sentences

    def getVerdict(self, score):
        if score < self.threshold:
            return "This text is most likely written by an Human"
        else:
            return "This text is most likely generated by an A.I."

    def getScore(self, sentence):
        import gc
        
        original_sentence = sentence
        sentence_length = len(list(re.finditer("[^\d\W]+", sentence)))
        # 减少生成的句子数量以节省内存
        remaining = 50
        sentences = self.mask(original_sentence, original_sentence, n=50, remaining=remaining)

        # 使用torch.no_grad()减少内存使用
        with torch.no_grad():
            real_log_likelihood = self.getLogLikelihood(original_sentence)

            generated_log_likelihoods = []
            # 分批处理生成的句子以减少内存占用
            batch_size = 5
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                for sentence in batch:
                    try:
                        log_likelihood = self.getLogLikelihood(sentence).cpu().detach().numpy()
                        generated_log_likelihoods.append(log_likelihood)
                    except Exception as e:
                        print(f"计算句子对数似然时出错: {e}")
                
                # 每处理一批后清理内存
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        if len(generated_log_likelihoods) == 0:
            return -1

        generated_log_likelihoods = np.asarray(generated_log_likelihoods)
        mean_generated_log_likelihood = np.mean(generated_log_likelihoods)
        std_generated_log_likelihood = np.std(generated_log_likelihoods)

        diff = real_log_likelihood - mean_generated_log_likelihood

        score = diff/(std_generated_log_likelihood)
        
        # 清理内存
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        return float(score), float(diff), float(std_generated_log_likelihood)

    def call_1_1(self, sentence, chunk_value):
        import gc
        
        sentence = re.sub("\[[0-9]+\]", "", sentence) # remove all the [numbers] cause of wiki

        words = re.split("[ \n]", sentence)

        # if len(words) < 100:
        #   return {"status": "Please input more text (min 100 words)"}, "Please input more text (min 100 characters)", None

        groups = len(words) // chunk_value + 1
        lines = []
        stride = len(words) // groups + 1
        for i in range(0, len(words), stride):
            start_pos = i
            end_pos = min(i+stride, len(words))

            selected_text = " ".join(words[start_pos:end_pos])
            selected_text = selected_text.strip()
            if selected_text == "":
                continue

            lines.append(selected_text)

        # sentence by sentence
        offset = ""
        scores = []
        probs = []
        final_lines = []
        labels = []
        
        # 处理较短的文本块以减少内存使用
        max_lines_per_batch = 3
        for batch_start in range(0, len(lines), max_lines_per_batch):
            batch_end = min(batch_start + max_lines_per_batch, len(lines))
            batch_lines = lines[batch_start:batch_end]
            
            for line in batch_lines:
                if re.search("[a-zA-Z0-9]+", line) == None:
                    continue
                try:
                    score, diff, sd = self.getScore(line)
                    if score == -1 or math.isnan(score):
                        continue
                    scores.append(score)

                    final_lines.append(line)
                    if score > self.threshold:
                        labels.append(1)
                        prob = "{:.2f}%\n(A.I.)".format(normCdf(abs(self.threshold - score)) * 100)
                        probs.append(prob)
                    else:
                        labels.append(0)
                        prob = "{:.2f}%\n(Human)".format(normCdf(abs(self.threshold - score)) * 100)
                        probs.append(prob)
                        
                except Exception as e:
                    print(f"处理文本块时出错: {e}")
                    continue
            
            # 每批次处理完毕后清理内存
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            #print(f"已处理 {batch_end}/{len(lines)} 个文本块")

        if not scores:
            return {"prob": "0.00%", "label": 0}, 0, "无法判断"
            
        mean_score = sum(scores)/len(scores)

        mean_prob = normCdf(abs(self.threshold - mean_score)) * 100
        label = 0 if mean_score > self.threshold else 1
        print(f"probability for {'A.I.' if label == 0 else 'Human'}:", "{:.2f}%".format(mean_prob))
        
        # 最后清理内存
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return {"prob": "{:.2f}%".format(mean_prob), "label": label}, mean_score, self.getVerdict(mean_score)

    def calculate_perplexity_api(self,text,api_model="gpt-4o"):
        """使用API计算文本的困惑度
        参数:
            text: 要分析的文本
            api_model: 选择使用的API模型，'gpt4o'或'deepseek'
        """
        import openai
        OPENAI_API_KEY = "sk-QXFhTtcE8Eo9NpA47501AdC0Da3840Ee96DfDc693400B9Ec"
        if api_model == "gpt-4o":
            # 使用OpenAI GPT-4o API
            client = openai.OpenAI(api_key=OPENAI_API_KEY,base_url="https://apione.zen-x.com.cn/api/v1")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": text}],
                temperature=0,
                logprobs=True,
            )
            
            # 获取log概率
        logprobs = [token.logprob for token in response.choices[0].logprobs.content]
        print(np.mean(logprobs))
        return(np.mean(logprobs))
    def getLogLikelihood(self,sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)

                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        return -1 * torch.stack(nlls).sum() / end_loc

################################################
#  Version 1 apis
###############################################

    def call_1(self, sentence):
        """
        Takes in a sentence split by full stop
        and print the perplexity of the total sentence
        split the lines based on full stop and find the perplexity of each sentence and print
        average perplexity
        Burstiness is the max perplexity of each sentence
        """
        results = OrderedDict()

        total_valid_char = re.findall("[a-zA-Z0-9]+", sentence)
        total_valid_char = sum([len(x) for x in total_valid_char]) # finds len of all the valid characters a sentence

        # if total_valid_char < 100:
        #    return {"status": "Please input more text (min 100 characters)"}, "Please input more text (min 100 characters)"

        lines = re.split(r'(?<=[.?!][ \[\(])|(?<=\n)\s*',sentence)
        lines = list(filter(lambda x: (x is not None) and (len(x) > 0), lines))

        ppl = self.getPPL_1(sentence)
        print(f"Perplexity {ppl}")
        results["Perplexity"] = ppl

        offset = ""
        Perplexity_per_line = []
        for i, line in enumerate(lines):
            if re.search("[a-zA-Z0-9]+", line) == None:
                continue
            if len(offset) > 0:
                line = offset + line
                offset = ""
            # remove the new line pr space in the first sentence if exists
            if line[0] == "\n" or line[0] == " ":
                line = line[1:]
            if line[-1] == "\n" or line[-1] == " ":
                line = line[:-1]
            elif line[-1] == "[" or line[-1] == "(":
                offset = line[-1]
                line = line[:-1]
            ppl = self.getPPL_1(line)
            Perplexity_per_line.append(ppl)
        print(f"Perplexity per line {sum(Perplexity_per_line)/len(Perplexity_per_line)}")
        results["Perplexity per line"] = sum(Perplexity_per_line)/len(Perplexity_per_line)

        print(f"Burstiness {max(Perplexity_per_line)}")
        results["Burstiness"] = max(Perplexity_per_line)

        out, label = self.getResults_1(results["Perplexity per line"])
        results["label"] = label

        return results, out

    def getPPL_1(self,sentence):
        encodings = self.tokenizer(sentence, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)

        nlls = []
        likelihoods = []
        prev_end_loc = 0
        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len
                likelihoods.append(neg_log_likelihood)

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        ppl = int(torch.exp(torch.stack(nlls).sum() / end_loc))
        return ppl

    def getResults_1(self, threshold):
        if threshold < 60:
            label = 0
            return "The Text is generated by AI.", label
        elif threshold < 80:
            label = 0
            return "The Text is most probably contain parts which are generated by AI. (require more text for better Judgement)", label
        else:
            label = 1
            return "The Text is written by Human.", label