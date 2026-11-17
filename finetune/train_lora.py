import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

name = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)

ds = load_dataset("json", data_files={"train": "train.jsonl", "validation": "val.jsonl"})

lora = LoraConfig(
    r=8,                 # size of the adapter matrices
    lora_alpha=16,        # how strongly the adapter affects the output (usually 2×r)
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",   # attention layers
                    "gate_proj", "up_proj", "down_proj"],     # MLP layers
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="out-lora",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,   # effective batch = 8 × 2 = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="no",
    bf16=True,
    max_length=1024,
    report_to="none",
)

trainer = SFTTrainer(
    model=model, args=args,
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    processing_class=tok, peft_config=lora,
)
trainer.model.print_trainable_parameters()
trainer.train()
trainer.save_model("out-lora/final")