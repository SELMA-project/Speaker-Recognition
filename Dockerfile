FROM python:3.9-slim

ARG REVISION=unknown
LABEL Revision=$REVISION

RUN export DEBIAN_FRONTEND=noninteractive \
  && apt-get -qq update \
  && apt-get -qq upgrade \
  && apt-get -qq install --no-install-recommends \
    g++ \
  && rm -rf /var/lib/apt/lists/*

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq dist-upgrade \
    && apt-get -qq install -y --no-install-recommends \
        ffmpeg


WORKDIR /data
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install nemo-toolkit[asr]==1.17.0

COPY diarization diarization
COPY embeddings embeddings
COPY models models
COPY examples examples
COPY retrain_classifier retrain_classifier
COPY SpeakerRecWebservice.py ./
COPY StartWebService.sh ./

EXPOSE 9002/tcp

ENTRYPOINT ["./StartWebService.sh"]
