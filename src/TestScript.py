import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

#MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"


# 4-bit settings: nf4 is the standard 4-bit format, double quant saves a bit
# more memory, and the actual math runs in bfloat16.
quant = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

start = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant,
    device_map={"": 0},  # put the whole model on GPU 0
    dtype=torch.bfloat16,
)
print(f"Loaded in {time.time() - start:.1f}s")
print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

messages = [
    {"role": "user", "content": "Write a Python function that reverses a string. Code only."}
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)

output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
new_tokens = output[0, inputs["input_ids"].shape[1] :]  # drop the prompt, keep the reply
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
