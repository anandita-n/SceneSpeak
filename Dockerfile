# SceneSpeak — CPU-only image (works on Hugging Face Spaces' free tier).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    PORT=7860 \
    VERIFIER=none

# Hugging Face Spaces runs containers as uid 1000.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

# CPU-only PyTorch first (the default wheel bundles CUDA and is several GB
# larger); the torch==2.4.1 pin in requirements.txt is then already satisfied.
COPY requirements.txt .
RUN pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY --chown=app:app . .
USER app

# Bake the slow first-run work into the image, so the container starts
# serving quickly: BLIP captioner weights, WordNet, and the RAG index
# (which also downloads Chroma's embedding model).
RUN python -c "from transformers import BlipForConditionalGeneration, BlipProcessor; \
m = 'Salesforce/blip-image-captioning-base'; BlipProcessor.from_pretrained(m); BlipForConditionalGeneration.from_pretrained(m)" \
 && python -c "from app.verification import _wordnet; _wordnet()" \
 && python -c "from app.rag import _get_collection; _get_collection()"

EXPOSE 7860
# Plain uvicorn (no --reload); PORT is set by the platform or defaults to 7860.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
