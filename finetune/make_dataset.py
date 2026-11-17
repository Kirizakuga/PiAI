import json, random

random.seed(0)
SYSTEM = "You are the friendly assistant on Karus's portfolio site. Keep answers short and casual."

text = open("../karus_data.txt", encoding="utf-8").read()
convos = []
for block in text.split("<eos>"):
    msgs = [{"role": "system", "content": SYSTEM}]
    for line in block.strip().split("\n"):
        if line.startswith(("User: ", "Q: ")):
            msgs.append({"role": "user", "content": line.split(": ", 1)[1]})
        elif line.startswith(("Assistant: ", "A: ")):
            msgs.append({"role": "assistant", "content": line.split(": ", 1)[1]})
    if len(msgs) > 1:                      # skip blocks with no Q/A (the "About Karus" bios)
        convos.append({"messages": msgs})

random.shuffle(convos)
convos = convos[:3000]                     # see note below
split = int(len(convos) * 0.95)
for name, part in [("train.jsonl", convos[:split]), ("val.jsonl", convos[split:])]:
    with open(name, "w", encoding="utf-8") as f:
        for c in part:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(len(convos), "conversations")