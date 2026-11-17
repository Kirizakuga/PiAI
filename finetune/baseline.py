import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda")

for q in ["who is Karus?", "what OS does Hiếu use?", "what is the capital of France?"]:
    messages = [{"role": "user", "content": q}]
    inputs = tok.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7)
    print("Q:", q)
    print("A:", tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True), "\n")