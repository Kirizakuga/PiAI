# PiAI: building Kou

Kou is a small chatbot that answers questions about Karus. It is Qwen2.5-0.5B-Instruct with a LoRA adapter trained on his own data, served on Azure Container Apps.

- **Talk to Kou:** the chat bubble on [kou.is-a.dev](https://kou.is-a.dev)
- **Devlog page:** [kou.is-a.dev/project/piai](https://kou.is-a.dev/project/piai/)

## Repo layout

| Path | What it is |
|---|---|
| `tried_from_scratch/` | Stage 1: a 12M-parameter GPT trained from scratch with nanoGPT |
| `finetune/make_dataset.py` | Builds the chat dataset (`train.jsonl`, `val.jsonl`) |
| `finetune/baseline.py` | Tests plain Qwen before fine-tuning |
| `finetune/train_lora.py` | Trains the LoRA adapter |
| `finetune/chat_lora.py` | Local chat with the base model + adapter |
| `finetune/merge.py` | Merges the adapter into Qwen for GGUF conversion |
| `finetune/system_prompt.txt` | Facts and rules, used in training and in chat |
| `finetune/app.py` | FastAPI server (`GET /health`, `POST /chat`) on llama-cpp-python |
| `finetune/Dockerfile` | The image deployed to Azure |

Model files (`*.gguf`, `*.safetensors`, checkpoints) are not in Git. The GGUF ships inside the Docker image on `ghcr.io/kirizakuga/kou`.

---

# Devlog

*Nov 18, 2026 · @kou*

## Summary

In about 7 hours (16:32 to 23:26, Tue 17 Nov 2026) Karus went from an empty whiteboard to Kou: a portfolio chatbot that is now live on the internet. It knows his facts, has its own name and personality, refuses to rate him and brushes off off-topic questions.

The road had three stages. First, a 12.25M-parameter GPT trained from scratch with nanoGPT (val loss 1.633); it only worked as a lookup table. Second, a LoRA fine-tune of Qwen2.5-0.5B-Instruct, improved over four data rounds. Third, shipping: merged into a 500 MB GGUF, packed in a Docker image and deployed to Azure Container Apps.

Training ran on an RTX 4050 laptop GPU (6 GB VRAM). Kou now runs on 1 CPU core in Azure's Malaysia West region, covered by the Azure for Students credit, so it cannot generate a bill. The code lives in this repo.

## Timeline

All times are Tue 17 Nov 2026, Bangkok time (UTC+7); ~ marks an approximate time. Each row is what happened, what went wrong, and how we got past it.

### Part 1: a tiny GPT from scratch

| Time | Step | Problem we faced | How we got past it |
|---|---|---|---|
| 16:38 | Step 1: data | A 4k-vocab model needs about 20 MB of text; one person's real writing is about 50 KB | Mixed TinyStories (general English) with text about Karus; wrote `prepare_data.py` |
| 16:51 | `karus_data.txt` v1 (5 MB, ~12,900 chats) | 5 MB from ~15 facts means heavy repetition; no invented facts allowed | Expanded each fact into many Q&A phrasings, plus "I don't know" answers for unshared facts |
| 16:54 | Tone fix | Answers sounded like "AI slop": emojis, "Pretty cool, right?" | Rewrote every answer short and casual, no emojis (~15,500 chats) |
| 17:15–17:38 | Built `corpus.txt` | 45.6 MB instead of ~25 MB: three separate bugs (next rows) | Checked doc counts against expected numbers each run |
| ↳ | Bug: edits not running | PyCharm ran `test.py` while the edits were elsewhere / unsaved; uploads were byte-identical | Compared the uploaded files and the run command line |
| ↳ | Bug: `<eos>` deleted | `clean()` strips anything shaped like an HTML tag, including `<eos>` | Split on `<eos>` first, then clean each piece |
| ↳ | Bug: data repeated 5× | `MY_TEXT_REPEAT` was still 5 (99,993 = 22,473 + 15,504 × 5) | Set it to 1: 37,977 docs, ~25 MB |
| 17:31–17:39 | Trained a 4k BPE tokenizer, encoded the corpus | The `tok` folder seemed missing; Vietnamese names showed as "LÃª" | Reload from Disk; explained byte-level display; `decode()` round-tripped perfectly. Result: 6.47M tokens |
| 17:48–17:51 | Set up nanoGPT | `FileNotFoundError` on `meta.pkl`: wrong folder (`karus/` vs `data/karus/`) and a leading "/" that meant the drive root | Moved the folder; used a full path |
| ~17:52–18:00 | Trained on the RTX 4050 | `torch.compile` doesn't work on Windows | `compile = False`, bfloat16. 12.25M params, ~6 min, best val loss 1.633 at step 4,750 |
| 18:02 | Uploaded the checkpoint | `ckpt.pt` sent as text came out garbled | Read the config out of the fragments, then asked for the training log |
| 18:07 | Chatted with the nanoGPT model | Answered "who is Kou?" as Karus, didn't know "hieu", no memory, random answers to off-topic questions | Diagnosed: a small model plus repetitive data acts as a lookup table. Chose to fine-tune a pretrained model |

### Part 2: housekeeping and Git

| Time | Step | Problem we faced | How we got past it |
|---|---|---|---|
| 18:21 | `git commit` | nanoGPT had its own `.git`, so it showed up as a submodule | Deleted the inner `.git`, `git rm --cached -f`; `.gitignore` for `*.pt`, `*.bin`, `.venv` |
| 18:24 | `git push` | "src refspec main does not match any" | The local branch was `master`: `git branch -M main` |
| 18:26 | Overwrote the old repo | The remote held an unrelated old project | Backed it up to an `old-backup` branch, then `push --force` |

### Part 3: fine-tuning Qwen into Kou

| Time | Step | Problem we faced | How we got past it |
|---|---|---|---|
| 18:31 | Plan: LoRA on Qwen2.5-0.5B-Instruct | Qwen3 has a thinking mode that complicates the chat format | Picked Qwen2.5 for a simpler first fine-tune |
| 19:18 | First run | "Torch not compiled with CUDA enabled": the new venv got CPU-only PyTorch | Reinstalled from the cu128 index: `2.11.0+cu128 True` |
| 19:19–19:20 | `baseline.py` | `apply_chat_template` now returns a dict, so `generate()` crashed | `return_dict=True` and `generate(**inputs)` |
| 19:21 | Baseline test | Plain Qwen claimed Hiếu is an Alibaba model running TensorFlow 2.4 | Saved as the "before" picture of hallucination |
| 19:24 | `train_lora.py` | `warmup_ratio` was removed from the library | `warmup_steps=10` |
| 19:38 | LoRA v1 (3,000 chats, r=16, 1 epoch, 7 min) | Eval loss 0.205, but it copied sentences, had no name of its own, and treated the weather as a question about Karus | Root cause was the data: gave the bot a name (Kou), an off-topic policy and "Karus hasn't told me" answers |
| 19:51–19:55 | v2 (1,000 chats, r=8, 2 epochs) | Name and follow-ups fixed; invented "Definitely" for teammate; an empty message made it claim to be Karus | Skipped empty input; planned opinion data |
| 20:00–20:01 | v3 (opinion, "are you Karus?", plain names) | hieu fixed, but it said Karus is "unpredictable under pressure" | Learned the limit: fine-tuning teaches style, not rules |
| 20:02–20:21 | v4: facts + rules system prompt, 233 opinion examples | System prompt (~330 tokens) made training 15 min and loss look artificially low (0.171) | `max_length=1024`; `system_prompt.txt` as the single source of truth |
| 20:25 | Final test | Only gap: "tell me more about him" | Rules generalized ("is he creative?"), injection resisted. Ready to deploy |

### Part 4: shipping Kou

| Time | Step | Problem we faced | How we got past it |
|---|---|---|---|
| ~20:30 | Merged the LoRA into Qwen (`merge.py`) | `kou-merged` not found, then an HF 401 error: the script ran from the wrong folder | `Path(__file__).parent` and `local_files_only=True` |
| ~20:40 | Converted to GGUF with llama.cpp | llama.cpp's `requirements.txt` would swap CUDA PyTorch for a CPU build; sentencepiece was missing | Installed only gguf-py + sentencepiece. Result: `kou-q8_0.gguf`, ~500 MB |
| ~20:50 | Wrote `app.py`: FastAPI + llama-cpp-python | A public endpoint needs abuse protection | Per-IP 2 s rate limit, last 7 messages × 500 chars, client system role dropped, CORS |
| ~21:00–21:30 | Searched for free hosting | HF Docker Spaces now paid; Oracle Always Free signup refused; Google Cloud Run billable after 90 days; Cloudflare Workers AI can't run a custom Qwen LoRA | Azure for Students: $100/year credit, no card, so it can't bill |
| 21:46 | `docker build` again | No prebuilt llama-cpp-python wheel for 0.3.37 on Linux; the slim image has no gcc | Added `build-essential`, plus `GGML_NATIVE=OFF` so the binary runs on Azure's CPUs |
| 22:12 | Pushed to GitHub Container Registry | Layers uploaded, then "denied" on the manifest; the package was left with no tag | Deleted the half-made package, logged in with a classic token (`write:packages`), pushed v1 |
| 22:23 | First deploy, Southeast Asia | `RequestDisallowedByAzure`: the student subscription allows only 5 regions | Read the region policy; picked Malaysia West, the closest allowed |
| 22:28–22:33 | Deployed to Container Apps | Defaults (up to 10 replicas, a Log Analytics workspace) could slowly eat the credit | Consumption plan, no logs, 1 CPU / 2 GiB, min 0 / max 1 replicas. `/chat` returned 200: "Hi, what's up?" |
| 23:23 | Widget live on the portfolio | Kou called itself "Kou Thi Quan Hieu" and claimed to be good at weather | Added a "your name is just Kou" prompt line; temperature 0.4 → 0.2 |
| 23:26 | Pushed the fix as a new image tag | Typo `gchr.io` and an extra argument to `docker push` | `ghcr.io`, one image name per push |

## How Kou works

Kou is three parts working together: a pretrained model that already knows English, a LoRA adapter that gives it Kou's voice and behaviour, and a system prompt that hands it Karus's facts and rules on every message.

```
runs once:              Qwen2.5-0.5B (frozen) + LoRA adapter → Kou
runs for every message: system_prompt.txt + last 3 exchanges + your message → chat template → Kou → reply
```

The adapter is about 4.4M parameters (under 20 MB on disk) and is loaded on top of the frozen ~1 GB Qwen model.

- **Base model:** Qwen2.5-0.5B-Instruct (490M parameters). It is never changed, which is why Kou still knows Paris is the capital of France.
- **LoRA adapter** (r=8, on all attention and MLP layers). Small trainable matrices added beside the frozen weights. Trained on 950 synthetic conversations for 2 epochs. It teaches behaviour: the name Kou, the casual tone, sassy redirects, refusing to rate Karus.
- **`system_prompt.txt`.** The facts list and three rules, prepended to every conversation, in training and in chat alike (one file, so they never drift apart). It carries the facts and rules, because fine-tuning alone didn't reliably enforce them (the v3 "unpredictable under pressure" answer).
- **`chat_lora.py`.** Skips empty input, keeps the system prompt plus the last 3 exchanges (so follow-ups like "i mean OS" work), formats them with Qwen's chat template, and samples a reply at temperature 0.4.

In production the same pieces are packed for a CPU server. One chat message travels like this:

1. A visitor opens the portfolio. The widget pings `GET /health` so Kou wakes from zero while they read (a cold start is roughly 20–60 s, an estimate).
2. The widget sends the conversation to `POST /chat` on Azure Container Apps (Malaysia West, 1 CPU, 2 GiB, at most 1 replica).
3. `app.py` (FastAPI) checks the 2 s rate limit, keeps the last 7 user/assistant messages and adds `system_prompt.txt`.
4. llama-cpp-python runs `kou-q8_0.gguf` (the LoRA merged into Qwen and stored in 8-bit) with the chat template inside the GGUF, up to 100 tokens.
5. The reply comes back as `{"reply": …}`. After 5 quiet minutes Kou scales back to zero and stops using the free grant.

Updating Kou means building a new Docker image, pushing it to `ghcr.io/kirizakuga/kou` with a new tag, and switching the tag in Azure; rolling back is switching the tag back.

## Results

Kou v4 passes every test except "tell me more about him". Each version fixed the previous version's worst failure.

| Test question | nanoGPT (12M, scratch) | Qwen 0.5B, no fine-tune | Kou v1 | Kou v2 | Kou v3 | Kou v4 |
|---|---|---|---|---|---|---|
| who is Karus / hieu? | ✅ Karus · ❌ hieu unknown | ❌ invented an Alibaba AI | ✅ | ⚠️ garbled for hieu | ✅ | ✅ |
| what's your name? | – | – | ❌ answered with Karus's name | ✅ Kou | ✅ | ✅ |
| follow-up ("i mean OS") | ❌ no memory | – | – | ✅ | ✅ | – |
| weather / Paris | ❌ random fact about Karus | ✅ Paris | ❌ "he hasn't told me" | ✅ redirect | ✅ | ✅ |
| good teammate? | – | – | – | ❌ "Definitely" (made up) | ❌ "unpredictable under pressure" | ✅ points to his projects |
| is he creative? (unseen) | – | – | – | – | ⚠️ invented details | ✅ "hasn't told me how to rate him" |
| prompt injection | – | – | – | – | – | ✅ resisted 2 attempts |
| tell me more about him | – | – | – | – | – | ❌ no summary in the data |

– = not tested in that round.

**Training cost:** nanoGPT ~6 min (5,000 steps) · Kou v1 7 min · v2 and v3 under 3 min each · v4 15 min (longer sequences).

**Running cost:** an estimated $0 a year up to about 500 chat visits a month, inside Container Apps' free grant. The worst case is the $100 credit running out and Kou going offline; there is no card, so never a bill.

**First live bugs:** on the real widget Kou mixed its name with Karus's ("Kou Thi Quan Hieu") and said it was "pretty good" at weather. The fix, a name line in the prompt and temperature 0.2, needs no retraining.

## Lessons learned and next steps

> The biggest lesson: **the data decides the behaviour.** Every real fix after 18:07 was a data change, not a model or code change.

- **Check numbers against expectations.** The corpus bugs were caught because 99,993 docs ≠ 37,977 and 11.7M tokens ≠ 6–8M.
- **Small from-scratch models memorize; pretrained models generalize.** nanoGPT learned sentences, Qwen learned behaviour.
- **Loss is not quality.** Loss went 0.205 → 0.33 → 0.17 while quality went bad → better → best. Only chat tests tell the truth.
- **Fine-tuning teaches style; the system prompt carries facts and rules.** Use both.
- **Libraries change.** transformers renamed `torch_dtype`, changed `apply_chat_template` and dropped `warmup_ratio`. Read the error, then the docs.
- **Never commit big generated files.** GitHub rejects files over 100 MB, so `.gitignore` `*.pt`, `*.bin`, `*.safetensors` and `*.gguf`; the model ships inside the Docker image instead.
- **Never put secrets in a system prompt.** Assume users can extract it.
- **Free tiers change fast.** In 2026 HF Docker Spaces went paid and Oracle's free tier shrank; check current terms before planning around one.
- **Make the bill impossible by design.** No card on file, max 1 replica, scale to zero. Then forgetting to check can't cost money.
- **Build for the machine it will run on.** `GGML_NATIVE=OFF` avoids an "Illegal instruction" crash when a laptop-built binary meets a cloud CPU.
- **Student subscriptions limit regions.** Read the "Allowed resource deployment regions" policy before picking one.

### Next steps

- [x] Merge the LoRA adapter into Qwen and convert it to GGUF (q8_0)
- [x] Serve it behind a `/chat` API with rate limits and CORS
- [x] Deploy it for free on Azure Container Apps
- [x] Add the chat widget to the portfolio
- [ ] Confirm the name and temperature fix is live
- [ ] Set a $10 budget alert in Azure
- [ ] Add a "tell me about him" summary category and retrain
- [ ] Assistant-only loss, so the loss number is honest and training is faster
- [ ] Stream replies so the first words appear sooner
- [ ] Write a README with the before/after examples from this log
