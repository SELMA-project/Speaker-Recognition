FROM python:3.8-slim

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONIOENCODING=utf-8

RUN export DEBIAN_FRONTEND=noninteractive \
    && echo 'deb-src http://deb.debian.org/debian bullseye main' \
    >> /etc/apt/sources.list \
    && apt-get -qq update \
    && apt-get -qq upgrade \
    && apt-get -qq install -y --no-install-recommends \
      g++ \
      ffmpeg \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /data
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY embeddings embeddings
COPY examples examples
COPY final_results final_results
COPY retrain_classifier retrain_classifier
COPY SpeakerRecWebservice.py ./
COPY StartWebService.sh ./

EXPOSE 9002

ENTRYPOINT ["./StartWebService.sh"]
