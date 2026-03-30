# ─────────────────────────────────────────────────────────────
# Build stage — installation des dépendances Python
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Dépendances système requises par OpenCV et scikit-image
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────
# Runtime stage
# ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Libs système runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgl1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copie des packages Python depuis le builder
COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Copie du code source
COPY appStreamlit/ ./appStreamlit/

# Poids EfficientNetB0 cachés dans l'image pour éviter le
# téléchargement au démarrage (optionnel — supprimez si réseau dispo)
RUN python -c "from tensorflow.keras.applications import EfficientNetB0; EfficientNetB0(weights='imagenet', include_top=False, pooling='avg')" \
    || echo "Pre-download skipped — weights will be fetched at runtime"

# Port Streamlit
EXPOSE 3039

# Variables d'environnement Streamlit
ENV STREAMLIT_SERVER_PORT=3039 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "appStreamlit/app.py", \
     "--server.port=3039", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
