from detectgpt import GPT2PPLV2 as GPT2PPL
model = GPT2PPL()
sentence = '''Our second key contribution is to represent this volumetric warp efficiently, and compute it in real time. Indeed,
even a relatively low resolution, 2563 deformation volume
would require 100 million transformation variables to be
computed at frame-rate. Our solution depends on a combination of adaptive, sparse, hierarchical volumetric basis
functions, and innovative algorithmic work to ensure a realtime solution on commodity hardware. As a result, DynamicFusion is the first system capable of real-time dense reconstruction in dynamic scenes using a single depth camera
'''
length = len(sentence)
if length < 300:
    chunk_value = 50
elif length < 800:
    chunk_value = length // 6
elif length < 1500:
    chunk_value = length // 10
else:
    chunk_value = 150
print(model(sentence, chunk_value, "v1.1"))
print(len(sentence),chunk_value)    

   