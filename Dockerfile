FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libglib is the one system lib OpenCV-headless still wants on slim images.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY adsb_camera ./adsb_camera
COPY tools ./tools
COPY config.example.yaml ./config.example.yaml
RUN mkdir -p /app/data

EXPOSE 8090
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/api/state' % os.environ.get('ADSBCAM__SERVER__PORT','8090'), timeout=4)"

# config.yaml is optional: mount one at /app/config.yaml, or use env vars.
CMD ["python", "-m", "adsb_camera", "-c", "/app/config.yaml"]
