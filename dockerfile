FROM python:3.10.11-alpine

RUN apk update \
    && apk add --no-cache linux-headers gcc g++ \
    && rm -rf /var/cache/apk/*

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app
RUN pip install -r requirements.txt

COPY . /app/
ENTRYPOINT [ "python", "app.py" ]