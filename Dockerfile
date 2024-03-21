FROM python:3-slim

RUN mkdir /app
COPY moderations.py requirements.txt /app/
#RUN git clone https://github.com/matatonic/openedai-moderations /app --single-branch
WORKDIR /app
RUN pip install -r requirements.txt

CMD python moderations.py
