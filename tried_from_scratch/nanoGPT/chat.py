import torch
from tokenizers import ByteLevelBPETokenizer
from model import GPTConfig, GPT   # nanoGPT's own model.py

device = "cuda"
ckpt = torch.load("out-karus/ckpt.pt", map_location=device)
model = GPT(GPTConfig(**ckpt["model_args"]))   # rebuild the same architecture
model.load_state_dict(ckpt["model"])           # put the trained weights in
model.to(device).eval()                         # eval() turns dropout off

tok = ByteLevelBPETokenizer("../tok/vocab.json", "../tok/merges.txt")
eos_id = tok.token_to_id("<eos>")

@torch.no_grad()   # no training here, so don't track gradients (faster, less memory)
def reply(question, max_new=80, temperature=0.7, top_k=40):
    prompt = f"User: {question}\nAssistant:"
    x = torch.tensor([tok.encode(prompt).ids], device=device)
    out = []
    for _ in range(max_new):
        logits, _ = model(x[:, -256:])           # model sees at most 256 tokens
        logits = logits[:, -1, :] / temperature  # scores for the NEXT token only
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("inf")   # keep only the top 40 options
        probs = torch.softmax(logits, dim=-1)          # scores → probabilities
        nxt = torch.multinomial(probs, 1)              # pick one at random by probability
        if nxt.item() == eos_id:
            break
        out.append(nxt.item())
        x = torch.cat([x, nxt], dim=1)                 # add it and go again
        if "\nUser:" in tok.decode(out):
            break
    return tok.decode(out).split("\nUser:")[0].strip()


while True:
    q = input("You: ")
    if q.lower() in ("quit", "exit"):
        break
    print("Bot:", reply(q))