import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
SYSTEM = open("system_prompt.txt", encoding="utf-8").read().strip()

tok = AutoTokenizer.from_pretrained(name)
base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "out-lora/final").to("cuda").eval()

history = [{"role": "system", "content": SYSTEM}]
while True:
    q = input("You: ")
    if not q.strip():
        continue
    if q.lower() in ("quit", "exit"):
        break
    history.append({"role": "user", "content": q})
    inputs = tok.apply_chat_template(
        history, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to("cuda")
    out = model.generate(**inputs, max_new_tokens=100, do_sample=True, temperature=0.7)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print("Bot:", answer)
    history.append({"role": "assistant", "content": answer})
    history = history[:1] + history[-6:]    # keep system prompt + last 3 exchanges