FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu -r requirements.txt
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF"
COPY kou-q8_0.gguf .
COPY system_prompt.txt app.py ./
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]