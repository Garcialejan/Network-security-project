FROM python:3.10-slim
WORKDIR /app
COPY . /app

RUN apt update -y && \
    apt install awscli -y --no-install-recommends awscli && \
    rm -rf /var/lib/apt/lists/*
    
RUN apt-get update && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]