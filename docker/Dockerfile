FROM python:3.13-slim

WORKDIR /app

# Install system deps needed by python-magic and lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/base.txt requirements/base.txt
RUN pip install --no-cache-dir -r requirements/base.txt

COPY src/ src/
COPY config.yaml config.yaml

ENV ENV=prod
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.main"]
