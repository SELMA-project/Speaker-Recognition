#!/bin/bash
export PYTHONUNBUFFERED=TRUE

#start web service
uvicorn "SpeakerRecWebservice:app" --host "0.0.0.0" --port "9002"
