FROM python:3-slim

RUN apt update && apt install -y git git-lfs

RUN mkdir -p /app/repos
WORKDIR /app
RUN git clone https://huggingface.co/ifmain/moderation_by_embeddings /app/repos/moderation_by_embeddings
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .

RUN apt clean && rm -rf /var/lib/apt/lists/*
RUN python moderations.py --test-load
CMD python moderations.py
