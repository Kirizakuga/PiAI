import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16)
model = PeftModel.from_pretrained(base, "out-lora/final").merge_and_unload()   # bake the adapter in

model.save_pretrained("kou-merged")
tok.save_pretrained("kou-merged")
print("saved to kou-merged/")