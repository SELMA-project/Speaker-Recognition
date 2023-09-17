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
import yt_dlp

from diarization.src.diarization import Pipeline
from retrain_classifier.Retrain_classifier import Retrain_classifier
from nemo.collections.asr.models import EncDecSpeakerLabelModel

import asyncio

class Speaker_Diarization_Recognition:
    def __init__(self):
        
        self.embedding_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
        self.classifier_path = "./models/KNN_auto_classifier.pkl"
        self.classifier = pickle.load(open(self.classifier_path, 'rb'))
        self.novel_detection_model_path = "./models/Local_Outlier_Filter.pkl"
        self.novel_speaker_detector = pickle.load(open(self.novel_detection_model_path, 'rb'))
        self.rttm_path = str()
        self.temp_folder = "./temp_folder"
        os.makedirs(self.temp_folder, exist_ok = True)
        self.final_results = "./final_results"
        os.makedirs(self.final_results, exist_ok = True)
        self.speaker_recognition_results = os.path.join(self.final_results, "speaker_recognition_results.pkl")
        
    def Download_video_from_youtube(self, url):
    
        ydl_opts = {
                'format': 'm4a/bestaudio/best',
                # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
                'postprocessors': [{  # Extract audio using ffmpeg
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                }]
            }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            Video_info = ydl.download(url)

        audio_file_source = "./"
        audio_file = glob.glob("%s/*.wav"%audio_file_source)
        
        return audio_file[0]

    def Process_audio_file(self, audio_file):

        dest_audio_file = os.path.join(self.temp_folder, "test_audio_file.wav")
        dest_audio = AudioSegment.from_file(audio_file)
        dest_audio = dest_audio.set_frame_rate(16000)
        dest_audio = dest_audio.set_channels(1)
        dest_audio.export(dest_audio_file, format="wav")

        return dest_audio_file

    def Perform_diarization(self, wav_file):

        diarizer = Pipeline.init_from_wav(wav_file)
        self.rttm_path = diarizer.write_to_RTTM("./temp_results")

    def get_label(self, wav_file):

        embeddings = self.embedding_model.get_embedding(wav_file).squeeze().numpy()

        label = self.classifier.predict([embeddings])

        return label

    def Perform_speaker_recognition(self, wav_file):

        audio_file = AudioSegment.from_file(wav_file)

        recognized_list = []

        with open(self.rttm_path, 'r') as rt_dia:
            rt_dia_lines = rt_dia.readlines()

            i = 0
            for rt_lines in rt_dia_lines:
                recognized = {}
                rt_data = rt_lines.split()

                start_time = float(rt_data[3])
                end_time = start_time + float(rt_data[4])

                start_sample = int(start_time) * 1000 #convert to milliseconds
                end_sample = int(end_time) * 1000 #convert to milliseconds

                audio_chunk_file_name = "sample_"+str(i)+".wav"
                audio_chunk_file_name = os.path.join(self.temp_folder, audio_chunk_file_name)
                audio_chunk = audio_file[start_sample:end_sample]
                audio_chunk.export(audio_chunk_file_name, format="wav")

                if self.Novel_speaker_detection(audio_chunk_file_name) == -1:
                    label = ["Unknown Speaker"]
                else:
                    label = self.get_label(audio_chunk_file_name)

                i = i + 1
                recognized["start time"] = start_time
                recognized["end time"] = end_time
                recognized["speaker label"] = label[0]

                recognized_list.append(recognized)
        
        pickle.dump(recognized_list, open(self.speaker_recognition_results, 'wb'))

        return recognized_list
    
    def Extract_labels(self, wav_file):

        #audio_file = AudioSegment.from_file(wav_file)
        #print(self.Novel_speaker_detection(wav_file))
        if str(self.Novel_speaker_detection(wav_file)) == str(-1):
            label = "Unknown Speaker"
            return label
        else:
            label = self.get_label(wav_file)
            return label[0]
    
    def Novel_speaker_detection(self, wav_file):

        embeddings = self.embedding_model.get_embedding(wav_file).squeeze().numpy()

        novel = self.novel_speaker_detector.predict([embeddings])

        return novel[0]

    def Perform_dia_rec(self, wav_file):
        
        wav_file = self.Process_audio_file(wav_file)
        self.Perform_diarization(wav_file)
        dia_rec_info = self.Perform_speaker_recognition(wav_file)
        self.Delete_local_files()
        
        return dia_rec_info

    def Delete_local_files(self):

        audio_file_source = "./"
        audio_file = glob.glob("%s/*.wav"%audio_file_source)

        if len(audio_file) == 1:
            os.remove(audio_file[0])

        shutil.copy2(self.rttm_path, self.final_results)
        shutil.rmtree(self.temp_folder)
        shutil.rmtree("./temp_results")
        
app = FastAPI()
Dia_Rec = Speaker_Diarization_Recognition()

# Combining both yutube and audio
title = "German Bundestag Speaker Identification Service"

examples_audio = [
    ["./examples/tagesschau02092019.wav"],
]

examples_url = [
    ["https://www.youtube.com/watch?v=E1q0yIW0O74"],
]

def Perform_Audio_dia_rec(audio_file):

    json_data = Dia_Rec.Perform_dia_rec(audio_file)

    return json_data

def Perform_Youtube_dia_rec(url):
    
    wav_file = Dia_Rec.Download_video_from_youtube(url)
    json_data = Dia_Rec.Perform_dia_rec(wav_file)
    
    return json_data

async def Audio_Video_dia_rec(audio_file, url):

    if audio_file is not None:
        json_data = Perform_Audio_dia_rec(audio_file)
    else:
        json_data = Perform_Youtube_dia_rec(url)

    return json_data

Audio_Video = gr.Blocks()

with Audio_Video:
    gr.Markdown("### This system is trained over a pre-trained Nemo-TitaNet model and K-Nearest Neighbours model.")
    gr.Markdown("### Speaker Recognition Service for YouTube Videos or Audio files that are already downloaded from YouTube.\
                Provide either the YouTube url or the downloaded audio.")
    YouTube_url = gr.Textbox(label="Link to YouTube URL")
    audio_input = gr.inputs.Audio(label="Input Audio", type="filepath")
    submit_button = gr.Button("Submit", variant="primary")
    result_output = gr.JSON(label="Speaker information based on start and end time of segments")
    gr.Examples(examples=examples_url, inputs=YouTube_url)
    gr.Examples(examples=examples_audio, inputs=audio_input)
    submit_button.click(Audio_Video_dia_rec, inputs=[audio_input, YouTube_url], outputs=result_output)
    
Audio_Video.queue(concurrency_count=20)    
# Tryout audio snippets
title = "German Bundestag Speaker Identification Service"
description = "This system is trained over a pre-trained Nemo-TitaNet model."

audio_input = gr.inputs.Audio(label="Input Audio", type="filepath")
result_output = gr.outputs.Textbox(label="Speaker ID")

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
    fn=Dia_Rec.Extract_labels,
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
        accuracy = Retrain.Retrain_classifiers_for_zip_audio()
    else:
        accuracy = Retrain.Retrain_classifiers_for_single_audio()

    return f"Training is completed with accuracy score of {accuracy}%"

title = "Add New Speaker(s) to Speaker Identification Service"
description = "To add new speaker(s), upload a single audio file with minimum 2 mins (for better recognition) \
               along with the speaker label. If more speakers needs to be added, add a zip file containing \
               multiple audio files named with respective speaker labels"

audio_file_input = gr.inputs.Audio(label="Input Audio for training (minimum of 2 minutes duration)", type="filepath")
label_name = gr.inputs.Textbox(label="Speaker label for training")
audio_zip_file_input = gr.File(label="Zip file with audio files named with speaker labels")
result_output = gr.outputs.Textbox(label="Training Results")


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

App_Mount = gr.TabbedInterface([Audio_Video, io_audio_sample, add_new_speakers],
                               ["Speaker Dia Rec", "Infer Audio Sample", "Add New Speakers"],
                               title=title)

#App_Mount.queue()
CUSTOM_PATH = "/Speaker_ID"
#if __name__ == "__main__":
app = gr.mount_gradio_app(app, App_Mount, path=CUSTOM_PATH)

    

    
    
