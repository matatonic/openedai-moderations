FROM python:3-alpine
LABEL version="0.1.0"

RUN mkdir /app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY moderations.py .

CMD python moderations.py
