import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
from pydantic import BaseModel

HERE = Path(__file__).parent
SYSTEM = (HERE / "system_prompt.txt").read_text(encoding="utf-8").strip()
llm = Llama(
    model_path=str(HERE / "kou-q8_0.gguf"),
    n_ctx=2048,                      # max tokens of prompt + reply the model sees
    n_threads=os.cpu_count(),        # use all CPU cores
    verbose=False,
)

app = FastAPI(title="Kou API", docs_url=None, redoc_url=None, openapi_url=None)  # hide /docs in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://kou.is-a.dev",      # portfolio (GitHub Pages)
        "http://localhost:63342",    # JetBrains built-in preview
        "http://localhost:5500",     # VS Code Live Server
        "http://127.0.0.1:5500",
        "http://localhost:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class Msg(BaseModel):
    role: str
    content: str

class ChatIn(BaseModel):
    messages: list[Msg]

last_seen = {}

@app.post("/chat")
def chat(body: ChatIn, request: Request):
    ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    now = time.time()
    if now - last_seen.get(ip, 0) < 2:
        raise HTTPException(429, "Slow down a bit.")
    last_seen[ip] = now

    msgs = [m for m in body.messages if m.role in ("user", "assistant")][-7:]
    if not msgs or msgs[-1].role != "user":
        raise HTTPException(400, "Last message must be from the user.")
    history = [{"role": "system", "content": SYSTEM}] + [
        {"role": m.role, "content": m.content[:500]} for m in msgs
    ]

    out = llm.create_chat_completion(messages=history, max_tokens=100, temperature=0.2)
    return {"reply": out["choices"][0]["message"]["content"].strip()}

@app.get("/health")
def health():
    return {"ok": True}