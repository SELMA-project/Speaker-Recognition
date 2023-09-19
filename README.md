# Speaker Identification Webservice

The pipeline performs speaker diarization and recognition in a single command. Here, unknown speaker detection is also integrated.

- Speaker diarization system: VBxHMM based.
- Unknown speaker detection: Local outlier factor ([LOF](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.LocalOutlierFactor.html))
- Speaker identification system: K-Nearest Neighbors ([KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html))
               
## Dataset Information

Dataset is an inhouse dataset consisting of over 392 Bundestag speakers with varying speaker duration from 30 seconds to 45 minutes.

## Speaker Diarization

Script to perform speaker diarization on an input WAV file using the VBx diarization system. It first uses voice activity detection (VAD) to segment the input audio file, computes speaker embeddings using a pre-trained x-vector model, and applies agglomerative hierarchical clustering (AHC) with a VBx hidden Markov model (HMM) to perform speaker diarization on the input audio file. Please call this script from the main folder directly.

# Webservice Usage

For the front end of the webservice FastApi and Gradio is used.

## Terminal Usage
To run the webservice from the terminal along with a browser use below command:

```
uvicorn --host 0.0.0.0 --port 9002 SpeakerRecWebservice:app
```

## Webserive Usage

The webservice has three tabs in it and are as follows:

1. Speaker Dia Rec - Here the diarization can be performaed either on a youtube video by providing the youtube link, or on an audio file of a downloaded youtube video.

2. Infer Audio Sample - Here aduio chunks can be uploaded and speaker labels can be inferred.

3. Add New Speakers - Here new speakers can be added to the existing model. Either a single speaker or a set of speakers can be added.
    - Single Speaker: upload audio file and the speaker label to add a single speaker.
    - Multiple Speaker: upload a zip file with multiple audio files whose names are considered as the speaker labels. 

## Docker build and run
To run the webservice by building and running from docker:

```
docker build  .
```

Once container is built, execute the following command.

```
docker run -d -p 9002:9002 "container id"
```

The webservice will be accessible under: http://nm-harmonia:9002/Speaker_ID/
