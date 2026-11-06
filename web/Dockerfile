# SPDX-License-Identifier: Apache-2.0
#
# IronZero training / inference platform image.
# Build from the repository root:
#   docker build -f web/Dockerfile -t ironzero-platform .
# The build needs network access for pip (CPU PyTorch wheel). Pass your proxy:
#   docker build -f web/Dockerfile \
#     --build-arg HTTPS_PROXY=http://host.docker.internal:7897 \
#     --build-arg HTTP_PROXY=http://host.docker.internal:7897 \
#     -t ironzero-platform .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY web/requirements.txt /tmp/requirements.txt
ARG HTTPS_PROXY=""
ARG HTTP_PROXY=""
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
 && pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY src ./src
COPY web ./web
COPY configs ./configs

RUN mkdir -p /app/runs/platform

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "web.backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
