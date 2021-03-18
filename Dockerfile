FROM ubuntu:19.10

COPY ./src /src
COPY requirements.txt /requirements.txt

RUN apt-get update \
    && apt-get install python3-dev python3-pip -y \
    && pip3 install -r requirements.txt

ENV PYTHONPATH=/src
WORKDIR /src

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]