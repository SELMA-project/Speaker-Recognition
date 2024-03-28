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

        Enrollment_embeddings = pickle.load(open("./embeddings/nemo_large_model_embeddings.pkl", 'rb'))

        test_embedding = self.embedding_model.get_embedding(wav_file).squeeze().cpu().numpy()

        known_label_list = []

        for enroll_embeddings in Enrollment_embeddings:

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

    def delete_speaker(self, speaker_name):

        updated_embeddings = []

        for speaker_info in speakers:

            if speaker_name != speaker_info["label"]:

                updated_embeddings.append(speaker_info)

        pickle.dump(updated_embeddings, open("./embeddings/nemo_large_model_embeddings.pkl", 'wb'))

        return "Speaker deleted and models are updated!"

    def speaker_list_update(self):

        list_of_speakers = []

        speakers = pickle.load(open("./embeddings/nemo_large_model_embeddings.pkl", 'rb'))

        list_of_speakers = [speaker["label"] for speaker in speakers]

        list_of_speakers = list(set(list_of_speakers))

        return gr.Dropdown(choices=list_of_speakers)
        
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
def Speaker_classifier_retrain_single_audio(audio_file=None, label=str(), audio_zip_files=None):
    Retrain = Retrain_classifier(audio_file, label, audio_zip_files)
    
    Retrain.Retrain_classifiers_for_single_audio()

    return f"Training is completed and the models are updated"

def Speaker_classifier_retrain_zip_files(audio_zip_files=None, audio_file=None, label=str()):
    Retrain = Retrain_classifier(audio_file, label, audio_zip_files)
    
    Retrain.Retrain_classifiers_for_zip_audio()

    return f"Training is completed and the models are updated"


add_new_speakers = gr.Blocks()
with add_new_speakers:
    gr.Markdown("# Add New Speaker(s) to Speaker Identification Service")
    gr.Markdown("## To add new speaker(s), upload a single audio file with minimum 2 mins (for better recognition) \
               along with the speaker label.")
    audio_file_input = gr.Audio(label="Input Audio for training (minimum of 2 minutes duration)", type="filepath")
    label_name = gr.Textbox(label="Speaker label for training")
    submit_button_1 = gr.Button("Submit", variant="primary")
    training_output_1 = gr.Textbox(label="training status")
    submit_button_1.click(Speaker_classifier_retrain_single_audio, inputs=[audio_file_input, label_name], outputs=training_output_1)
    gr.Markdown("## If more speakers needs to be added, add a zip file containing \
               multiple audio files named with respective speaker labels. Just upload the zip file and models will be updated automatically.")
    zip_files = gr.File(label="Zip file with audio files named with speaker labels", file_types=["file"], type="filepath")
    training_output_2 = gr.Textbox(label="training status")
    zip_files.upload(Speaker_classifier_retrain_zip_files, inputs=zip_files, outputs=training_output_2)

# )

# Delete speaker tab
list_of_speakers = []

speakers = pickle.load(open("./embeddings/nemo_large_model_embeddings.pkl", 'rb'))

list_of_speakers = [speaker["label"] for speaker in speakers]

list_of_speakers = list(set(list_of_speakers))

delete_speaker_demo = gr.Blocks()

with delete_speaker_demo:
    gr.Markdown("# Delete speaker")
    gr.Markdown("## Delete a selected speaker")
    dropdown = gr.Dropdown(list_of_speakers, label="List of known speakers", info="Select a speaker to delete from the model")
    submit_button = gr.Button("Submit", variant="primary")
    training_output = gr.Textbox(label="training status")
    gr.Markdown("### Clear button updates the dropdown list")
    clear_button = gr.Button("Clear", variant="primary")
    submit_button.click(Spk_Rec.delete_speaker, inputs=dropdown, outputs=training_output)
    clear_button.click(Spk_Rec.speaker_list_update, outputs=dropdown)


# Gradio Fast api mount

#App_Mount = gr.TabbedInterface([io_youtube_audio, Youtube_video, io_audio_sample, add_new_speakers],
#                               ["Speaker Dia Rec", "YouTube Speaker Rec", "Infer Audio Sample", "Add New Speakers"])
title = "German Bundestag Speaker Identification Service"

App_Mount = gr.TabbedInterface([io_audio_sample, add_new_speakers, delete_speaker_demo],
                               ["Infer Audio Sample", "Add New Speakers", "Delete Speaker From Model"],
                               title=title)

#App_Mount.queue()
CUSTOM_PATH = "/Speaker_ID"
#if __name__ == "__main__":
app = gr.mount_gradio_app(app, App_Mount, path=CUSTOM_PATH)