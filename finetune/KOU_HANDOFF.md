# Handoff: add Kou chat widget to my portfolio

## Context
- I'm Karus (Lê Quán Hiếu), 2nd-year IT student at PTIT HCMC.
- **PiAI** is my project: a fine-tuned chatbot named **Kou** that answers visitors' questions about me.
  - Base model: Qwen2.5-0.5B-Instruct, plus a LoRA trained on my own data.
  - Merged and converted to GGUF (q8_0, ~500 MB).
  - Served by FastAPI + llama-cpp-python.
- Kou is already **deployed and working** on Azure Container Apps (Azure for Students, Malaysia West region).
- **Goal now:** add a chat widget to my portfolio site that talks to Kou.

### How I like to work
- Explain changes step by step so I understand them.
- Don't dump whole files on me without saying what each part does.
- Keep UI text natural and casual, not "AI-sounding".

## Repos
- PiAI (the model + API): `C:\Users\karus\Documents\project\PiAI`, GitHub `Kirizakuga/PiAI`.
  - The API code lives in `finetune/`: `app.py`, `system_prompt.txt`, `kou-q8_0.gguf`, `Dockerfile`, `requirements.txt`, `.dockerignore`.
- Portfolio site: **the repo we're working in now.** Check its framework/stack before suggesting anything.

## Kou API
Base URL: `https://kou.happypond-8b406318.malaysiawest.azurecontainerapps.io`

| Endpoint | Request | Response |
|---|---|---|
| `GET /health` | — | 200 when the server is up |
| `POST /chat` | `{"messages": [{"role": "user", "content": "hi"}, ...]}` | `{"reply": "Hi, what's up?"}` |

Server-side rules (already in `app.py`):
- Only `user` / `assistant` roles are kept. A client-sent `system` role is dropped (the server adds its own system prompt).
- The server keeps only the **last 7 messages**, each cut to **500 chars**.
- The last message must be `user`. Otherwise it returns **400**.
- Rate limit is **one request per 2 s per IP**. Over the limit returns **429**.
- Replies are short (max_tokens 100) and **not streamed**: the full reply comes back at once.

## Widget requirements
1. **Wake Kou on page load:** the app scales to zero when idle, so a cold start takes about 20–60 s.
   - Fire `GET /health` as soon as the portfolio loads (don't block on it), so Kou is awake by the time someone types.
2. **"Kou is waking up…" state:**
   - If a reply takes longer than ~3 s, show a friendly waiting message.
   - Use a generous fetch timeout (~90 s).
3. **Errors:**
   - 429 → "slow down a sec" style message.
   - Network error or 5xx → casual "Kou's napping, try again" message with a retry button.
4. **History:**
   - Keep the conversation client-side only (memory or sessionStorage).
   - Send the last ~6 messages with each request.
5. **Input:**
   - Trim input, block empty messages, cap length at 500 chars.
   - Disable the send button while waiting.
6. **Look and feel:**
   - Floating button in a corner that opens a small chat panel; must work on mobile.
   - Greeting: something like "Hey, I'm Kou — ask me anything about Karus."
   - Kou's vibe is short, casual and a bit sassy.
7. No API keys or secrets are needed. The endpoint is public by design.

## Required backend change: CORS
`app.py` currently allows only `http://localhost:3000`:
```python
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:3000"], allow_methods=["POST"], ...)
```
To do:
- Add the portfolio's real origin(s), e.g. `https://<my-domain>`, plus the local dev URL the portfolio actually uses.
- Allow `GET` too if the browser calls `/health` directly.
- Optional while we're in there: `FastAPI(docs_url=None, redoc_url=None)` to hide `/docs` in production.

### Redeploy flow (run from `PiAI/finetune/`)
```
docker build -t kou .
docker tag kou ghcr.io/kirizakuga/kou:v2
docker push ghcr.io/kirizakuga/kou:v2
```
Then in the Azure portal: container app `kou` → Application → Containers → Edit and deploy → change the image tag to `v2` → Create. Roll back by switching to `v1`.

## Don'ts (cost / safety)
- Keep **min replicas 0, max replicas 1** on Azure.
- Don't add Azure Container Registry. Images live on GHCR (public package `ghcr.io/kirizakuga/kou`).
- Don't add Log Analytics or any paid add-on.
- Don't send any secrets to the frontend.

## Ideas for later (not now)
- Streaming replies: would need a new server endpoint using llama-cpp's `stream=True` + Server-Sent Events.
- A "tell me about Karus" summary: Kou sometimes deflects that question; fix via a system-prompt line or more training data.
