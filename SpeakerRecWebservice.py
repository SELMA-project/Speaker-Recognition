#!/usr/bin/env python3
# coding: utf-8
#----------------------------------------------------------------------------
# Created By  : shikkalven
#----------------------------------------------------------------------------

import gradio as gr
from fastapi import FastAPI

import pydub
from pydub import AudioSegment
import os
import pickle
import shutil
import glob
import torch

from retrain_classifier.Retrain_classifier import Retrain_classifier
from nemo.collections.asr.models import EncDecSpeakerLabelModel

import asyncio

class Speaker_Recognition:
    def __init__(self):
        
        self.device = "cpu" #Device on which the computations happen
        self.embedding_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large", map_location=self.device)
        self.temp_folder = "./temp_folder"
        os.makedirs(self.temp_folder, exist_ok = True)
        self.final_results = "./final_results"
        os.makedirs(self.final_results, exist_ok = True)
        self.Enrollment_embeddings = pickle.load(open("./embeddings/nemo_large_model_embeddings.pkl", 'rb'))

    def Process_audio_file(self, audio_file):

        dest_audio_file = os.path.join(self.temp_folder, "test_audio_file.wav")
        dest_audio = AudioSegment.from_file(audio_file)
        dest_audio = dest_audio.set_frame_rate(16000)
        dest_audio = dest_audio.set_channels(1)
        wav_file_duration = int(dest_audio.duration_seconds)
        if wav_file_duration >= 5:
            dest_audio = dest_audio[0:5000]
        dest_audio.export(dest_audio_file, format="wav")

        return dest_audio_file

    def Perform_speaker_recognition(self, wav_file):

        label = self.Extract_labels(wav_file)

        return label
    
    def Extract_labels(self, wav_file, i=0):

        test_embedding = self.embedding_model.get_embedding(wav_file).squeeze().cpu().numpy()

        known_label_list = []

        for enroll_embeddings in self.Enrollment_embeddings:

            # Length Normalize
            X = torch.from_numpy(test_embedding) / torch.linalg.norm(torch.from_numpy(test_embedding))
            Y = torch.from_numpy(enroll_embeddings["embeddings"]) / torch.linalg.norm(torch.from_numpy(enroll_embeddings["embeddings"]))
            # Score
            similarity_score = torch.dot(X, Y) / ((torch.dot(X, X) * torch.dot(Y, Y)) ** 0.5)
            similarity_score = (similarity_score + 1) / 2

            if similarity_score >= 0.80:
                known_label_list.append(enroll_embeddings["label"])

        if len(known_label_list) == 0:
            label = "Unknown Speaker / Reporter / VoiceOver"
        else:
            label = known_label_list[0]
            if "Wolfganag" in label:
                label = label.replace("Wolfganag", "Wolfgang")
        
        return label

    def Perform_rec(self, wav_file):
        
        wav_file = self.Process_audio_file(wav_file)
        rec_info = self.Perform_speaker_recognition(wav_file)
        
        return rec_info
        
app = FastAPI()
Spk_Rec = Speaker_Recognition()

#Audio_Video.queue(concurrency_count=20)    
# Tryout audio snippets
title = "German Bundestag Speaker Identification Service"
description = "This system is trained over a pre-trained Nemo-TitaNet model."

audio_input = gr.Audio(label="Input Audio", type="filepath")
result_output = gr.Textbox(label="Speaker ID")

examples = [
    ["./examples/Olaf_Scholz.wav"],
    ["./examples/Klara_Geywitz.wav"],
    ["./examples/Dr.Karl_Lauterbach.wav"],
    ["./examples/Hubertus_Heil.wav"],
    ["./examples/Robert_Habeck.wav"],
    ["./examples/Steffi_Lemke.wav"],
    ["./examples/Wolfganag_Schmidt.wav"]
]

io_audio_sample = gr.Interface(
    fn=Spk_Rec.Perform_rec,
    title=title,
    description=description,
    examples=examples,
    inputs=audio_input,
    outputs=result_output,
    allow_flagging="never",
    css="footer {visibility: hidden}",
)

# Add new speakers
def Speaker_classifier_retrain(audio_file=None, label=str(), audio_zip_files=None):
    Retrain = Retrain_classifier(audio_file, label, audio_zip_files)
    if audio_zip_files is not None:
        Retrain.Retrain_classifiers_for_zip_audio()
    else:
        Retrain.Retrain_classifiers_for_single_audio()

    return f"Training is completed and the models are updated"

title = "Add New Speaker(s) to Speaker Identification Service"
description = "To add new speaker(s), upload a single audio file with minimum 2 mins (for better recognition) \
               along with the speaker label. If more speakers needs to be added, add a zip file containing \
               multiple audio files named with respective speaker labels"

audio_file_input = gr.Audio(label="Input Audio for training (minimum of 2 minutes duration)", type="filepath")
label_name = gr.Textbox(label="Speaker label for training")
audio_zip_file_input = gr.File(label="Zip file with audio files named with speaker labels")
result_output = gr.Textbox(label="Training Results")


add_new_speakers = gr.Interface(
    fn=Speaker_classifier_retrain,
    title=title,
    description=description,
    #article=article,
    #examples=examples,
    inputs=[audio_file_input, label_name, audio_zip_file_input],
    outputs=result_output,
    allow_flagging="never",
    css="footer {visibility: hidden}",
)

# Gradio Fast api mount

#App_Mount = gr.TabbedInterface([io_youtube_audio, Youtube_video, io_audio_sample, add_new_speakers],
#                               ["Speaker Dia Rec", "YouTube Speaker Rec", "Infer Audio Sample", "Add New Speakers"])
title = "German Bundestag Speaker Identification Service"

App_Mount = gr.TabbedInterface([io_audio_sample, add_new_speakers],
                               ["Infer Audio Sample", "Add New Speakers"],
                               title=title)

#App_Mount.queue()
CUSTOM_PATH = "/Speaker_ID"
#if __name__ == "__main__":
app = gr.mount_gradio_app(app, App_Mount, path=CUSTOM_PATH)