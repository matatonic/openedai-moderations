FROM python:3-alpine

RUN mkdir /app
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY *.py .

CMD python moderations.py
