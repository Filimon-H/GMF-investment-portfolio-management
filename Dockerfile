# ---- Base image ----
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Harden APT: use HTTPS mirrors (deb822 or classic), add retries/timeouts
RUN set -eux; \
    if [ -f /etc/apt/sources.list ]; then \
      sed -i 's|http://|https://|g' /etc/apt/sources.list; \
    fi; \
    if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
      sed -i -e 's|http://deb.debian.org|https://deb.debian.org|g' \
             -e 's|http://security.debian.org|https://security.debian.org|g' \
             /etc/apt/sources.list.d/debian.sources; \
    fi; \
    printf 'Acquire::Retries "5"; Acquire::http::Timeout "30"; Acquire::https::Timeout "30";\n' \
      > /etc/apt/apt.conf.d/80-retries; \
    apt-get update; \
    apt-get install -y --no-install-recommends build-essential curl git ca-certificates; \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Faster builds & fewer wheels issues
RUN python -m pip install --upgrade pip setuptools wheel

# Dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

# Streamlit defaults (can be overridden by env)
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.headless=true", "--server.port=8501", "--browser.gatherUsageStats=false"]
