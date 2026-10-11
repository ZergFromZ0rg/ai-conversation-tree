# --- build the viewer ---
FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- runtime ---
FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    AI_CONVERSATION_TREE_DB=/data/conversationTree.db \
    AI_CONVERSATION_TREE_STUB_LLM=1

COPY requirements.txt ./
# CPU-only torch: the default Linux wheel drags in ~2 GB of CUDA libraries.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Bake the two models into the image so the container starts offline. HF_HOME
# is a fixed, world-readable path because the app runs as an unprivileged user
# whose home directory is not the one the build used.
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
RUN chmod -R a+rX /opt/hf
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY conversation_tree ./conversation_tree
COPY eval_*.py eval_cases.json ./
COPY --from=frontend /app/frontend/dist ./frontend/dist

RUN mkdir /data && useradd --system app && chown app /data
USER app
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "conversation_tree.api:app", "--host", "0.0.0.0", "--port", "8000"]
