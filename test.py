from detectgpt import GPT2PPLV2 as GPT2PPL
import math
from scipy.stats import norm
def custom_piecewise_normalize_parameterized(score, threshold=0.2):
  """
  将输入值（预期在 -1 到 1 之间）根据阈值进行分段归一化：
  - 如果 score >= threshold，则线性映射到 [0.8, 1.0] 区间。
  - 如果 score < threshold，则线性映射到 [0, 0.2] 区间。

  Args:
    score: 输入的分数，预期在 -1 到 1 之间。
    threshold: 用于分段的阈值。函数假设 threshold 在 (-1, 1) 范围内
               以确保两个分段都有意义。

  Returns:
    归一化后的值，在 [0, 0.2] 或 [0.8, 1.0] 区间内。
    在 threshold 等于 -1 或 1 的极端情况下，行为可能简化。
  """
  # 定义输入和输出范围的边界
  input_min = -1.0
  input_max = 1.0
  output_lower_min = 0.0
  output_lower_max = 0.2
  output_upper_min = 0.8
  output_upper_max = 1.0

  # --- 处理 score >= threshold 的情况 ---
  if score >= threshold:
    # 输入范围是 [threshold, input_max] (长度 input_max - threshold)
    # 输出范围是 [output_upper_min, output_upper_max] (长度 output_upper_max - output_upper_min)

    # 计算输入范围的长度
    input_range_len = input_max - threshold
    # 计算输出范围的长度
    output_range_len = output_upper_max - output_upper_min

    # 处理边缘情况：如果 threshold 等于或大于 input_max，则输入范围长度为0或负
    # 此时所有 score >= threshold 的值都应映射到输出范围的上限
    if input_range_len <= 0:
        return output_upper_max

    # 计算 score 在输入范围内的相对位置（0 到 1 之间）
    # (score - 输入范围起点) / 输入范围长度
    relative_position = (score - threshold) / input_range_len

    # 将该相对位置应用到输出范围：输出范围起点 + 相对位置 * 输出范围长度
    normalized_value = output_upper_min + relative_position * output_range_len
    # 确保结果在 [0.8, 1.0] 区间内，防止浮点数精度问题导致略微超出
    return max(output_upper_min, min(normalized_value, output_upper_max))

  # --- 处理 score < threshold 的情况 ---
  else:
    # 输入范围是 [input_min, threshold) (长度 threshold - input_min)
    # 输出范围是 [output_lower_min, output_lower_max) (长度 output_lower_max - output_lower_min)

    # 计算输入范围的长度
    input_range_len = threshold - input_min
     # 计算输出范围的长度
    output_range_len = output_lower_max - output_lower_min

    # 处理边缘情况：如果 threshold 等于或小于 input_min，则输入范围长度为0或负
    # 此时所有 score < threshold 的值（如果存在且小于input_min）都应映射到输出范围的下限
    if input_range_len <= 0:
         return output_lower_min # 映射到0.0

    # 计算 score 在输入范围内的相对位置（0 到 1 之间）
    # (score - 输入范围起点) / 输入范围长度
    relative_position = (score - input_min) / input_range_len

    # 将该相对位置应用到输出范围：输出范围起点 + 相对位置 * 输出范围长度
    normalized_value = output_lower_min + relative_position * output_range_len
    # 确保结果在 [0, 0.2] 区间内，防止浮点数精度问题导致略微超出
    return max(output_lower_min, min(normalized_value, output_lower_max))
model = GPT2PPL()
sentence = '''RoBERT builds upon BERT's architecture but introduces several optimizations to improve
efficiency and performance:
Dynamic Masking : Unlike BERT, which uses static masking during pretraining, RoBERT
dynamically masks tokens for each epoch. This approach ensures that the model is
exposed to a wider variety of masked token patterns, enhancing its generalization
capabilities..
'''
length = len(sentence)
if length < 300:
    chunk_value = 50
elif length < 600:
    chunk_value = length // 6 
elif length < 800:
    chunk_value = length // 7 
elif length < 1000:
    chunk_value = length // 9
elif length < 1500:
    chunk_value = length // 10
else:
    chunk_value = 150
prob,score,_=model(sentence, chunk_value, "v1.1")
print(prob,score,custom_piecewise_normalize_parameterized(score),norm.cdf(score))
print(len(sentence),chunk_value)    

   