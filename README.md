# Speaker Identification Webservice

The pipeline performs speaker recognition in a single command. Here, unknown speaker detection is also integrated. Speaker recognition is performed based on threshold values based on cosine similarity between the enrollment embeddings and the test embeddings.

               
## Dataset Information

Dataset is an inhouse dataset consisting of over 392 Bundestag speakers with varying speaker duration from 30 seconds to 45 minutes.

Here we have selected 358 speakers based on the duration ranging from 10 mins of data to the maximum duration available in the dataset.

Dataset link: /nm-raid/audio/work/mturan/Bundestag_speaker_data/Recognition/

## List of Speakers
Here are the list of 358 speakers: [list of speakers](list_of_speakers.txt)

## Model Performance

Test set: is based on the video data collected for IBC demo.

Test set location: /nm-raid/audio/work/shikkalven/SELMA/SpkRecBenchmark/DE_TestSet

Test samples: 2751 test segments

Total Number of known speakers: 36

Model Accuracy: 96.55%



# Webservice Usage

For the front end of the webservice FastApi and Gradio is used.

## Terminal Usage
To run the webservice from the terminal along with a browser use below command:

```
uvicorn --host 0.0.0.0 --port 9002 SpeakerRecWebservice:app
```

## Webserive Usage

The webservice has two tabs in it and are as follows:

1. Infer Audio Sample - Here aduio chunks can be uploaded and speaker labels can be inferred.

2. Add New Speakers - Here new speakers can be added to the existing model. Either a single speaker or a set of speakers can be added.
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

The webservice can be accessed under: localhost:9002/Speaker_ID/
