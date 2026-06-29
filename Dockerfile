FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install deps, then force jinja2==3.1.2 — Gradio 4.44.1 was built against
# 3.1.x. In 3.1.3+ make_globals() passes a dict into the LRU cache key, which
# is unhashable and crashes every page render. 3.1.2 doesn't have that code path.
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "jinja2==3.1.2"

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
