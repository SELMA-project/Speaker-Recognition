#!/usr/bin/env python3
# coding: utf-8
#----------------------------------------------------------------------------
# Created By  : shikkalven
#----------------------------------------------------------------------------

import pydub
from pydub import AudioSegment
import os
import glob
import shutil
import pickle
import pandas as pd
import zipfile
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from nemo.collections.asr.models import EncDecSpeakerLabelModel

class Retrain_classifier:
    def __init__(self, audio_file, label, audio_zip_files):

        self.nemo_model = EncDecSpeakerLabelModel.from_pretrained(model_name="titanet_large")
        self.ML_embeddings = "./embeddings/nemo_large_model_embeddings.pkl"
        self.Embeddings_list = pickle.load(open(self.ML_embeddings, 'rb'))
        self.Retrain_folder = os.path.join("./retrain_classifier", "retrain_folder")
        os.makedirs(self.Retrain_folder, exist_ok = True)
        self.Retrain_folder_zip = os.path.join("./retrain_classifier", "retrain_folder_zip")
        os.makedirs(self.Retrain_folder_zip, exist_ok = True)
        self.wav_file = audio_file
        self.label = label.replace(" ", "_")
        self.wav_zip_files = audio_zip_files
        self.Novelty_detector = "./models/Local_Outlier_Filter.pkl"
        self.Speaker_classifier = "./models/KNN_auto_classifier.pkl"

    def Prepare_audio_chunks(self, audio_files):

        WINDOW_LENGTH_3sec = 3
        WINDOW_LENGTH_5sec = 5
        WINDOW_LENGTH_10sec = 10

        audio_file = AudioSegment.from_file(audio_files)
        wav_file_duration = int(audio_file.duration_seconds)

        i = 0
        t1 = 0
        t2 = 0
        WINDOW_LENGTH = 0
        WINDOW_LENGTH_SEC = 0

        if wav_file_duration >= 60: #checking if the audio length is atleast 1 min
            if wav_file_duration >= 300: #limiting audio length to 5 mins
                audio_file = audio_file[0:300000]
                wav_file_duration = int(audio_file.duration_seconds)

                while wav_file_duration > 0: #loop until audio file length is 5 seconds

                    if i < 30:
                        if i == 0:
                            t2 = WINDOW_LENGTH_3sec*1000
                        WINDOW_LENGTH = 3*1000
                        WINDOW_LENGTH_SEC = 3
                    elif i >= 30 and i < 50:
                        WINDOW_LENGTH = 5*1000
                        WINDOW_LENGTH_SEC = 5
                    else:
                        WINDOW_LENGTH = 10*1000
                        WINDOW_LENGTH_SEC = 10

                    new_file = audio_file[t1:t2]
                    audio_file_name = self.label+'_'+str(i)+'.wav'
                    audio_file_name = os.path.join(self.Retrain_folder, audio_file_name)
                    new_file.export(audio_file_name, format="wav")

                    t1 = t2
                    t2 = WINDOW_LENGTH + t2
                    i = i + 1
                    wav_file_duration = wav_file_duration - WINDOW_LENGTH_SEC

            else:
                while wav_file_duration > 0: #loop until audio file length is 5 seconds

                    if i < 10:
                        if i == 0:
                            t2 = WINDOW_LENGTH_3sec*1000
                        WINDOW_LENGTH = 3*1000
                        WINDOW_LENGTH_SEC = 3
                    elif i >= 10 and i < 16:
                        WINDOW_LENGTH = 5*1000
                        WINDOW_LENGTH_SEC = 5
                    else:
                        WINDOW_LENGTH = 10*1000
                        WINDOW_LENGTH_SEC = 10

                    new_file = audio_file[t1:t2]
                    audio_file_name = self.label+'_'+str(i)+'.wav'
                    audio_file_name = os.path.join(self.Retrain_folder, audio_file_name)
                    new_file.export(audio_file_name, format="wav")
                    t1 = t2
                    t2 = WINDOW_LENGTH + t2
                    i = i + 1
                    wav_file_duration = wav_file_duration - WINDOW_LENGTH_SEC 

    def Update_embeddings(self):

        new_audio_files = glob.glob("%s/*.wav"%self.Retrain_folder)

        for wav_file in new_audio_files:
            
            Embeddings_dictionary = {}
            label = wav_file.split("/")[-2]
            Embeddings_dictionary['label'] = label
            embeddings = self.nemo_model.get_embedding(wav_file).squeeze()
            Embeddings_dictionary['embeddings'] = embeddings.numpy()
            
            self.Embeddings_list.append(Embeddings_dictionary)

        pickle.dump(self.Embeddings_list, open(self.ML_embeddings, 'wb'))

    def Prepare_data_for_training(self):

        Embeddings_list = pickle.load(open(self.ML_embeddings, 'rb'))

        ML_embeddings_data = pd.DataFrame(Embeddings_list)
        ML_embeddings_data.columns = ["Speaker labels", "Speaker embeddings"]

        X_train, X_test, Y_train, Y_test = train_test_split(ML_embeddings_data, 
                                                    ML_embeddings_data['Speaker labels'],
                                                    test_size=0.9,
                                                    random_state=0)

        self.X_train = X_train.reset_index(drop=True)
        self.X_test = X_test.reset_index(drop=True)

    def Train_novelty_detector(self):

        novelty_detector = LocalOutlierFactor(n_neighbors=4, novelty=True)

        novelty_detector.fit(self.X_train["Speaker embeddings"].tolist(),
                             self.X_train['Speaker labels'].tolist())

        pickle.dump(novelty_detector, open(self.Novelty_detector, 'wb'))

    def Train_speaker_classifier(self):

        KNN_classifier_auto = KNeighborsClassifier(n_neighbors=4)

        KNN_classifier_auto.fit(self.X_train["Speaker embeddings"].tolist(),
                                self.X_train['Speaker labels'].tolist())

        pickle.dump(KNN_classifier_auto, open(self.Speaker_classifier, 'wb'))

        predictions = KNN_classifier_auto.predict(self.X_test["Speaker embeddings"].tolist())

        return np.round(accuracy_score(self.X_test['Speaker labels'], predictions)*100, 2)

    def Prepare_audio_chunks_of_zip_file(self):

        with zipfile.ZipFile(self.wav_zip_files, 'r') as zip_ref:
            zip_ref.extractall(self.Retrain_folder_zip)

        audio_files_from_zip = glob.glob("%s/*.wav"%self.Retrain_folder_zip)

        for audio_files in audio_files_from_zip:
            self.Prepare_audio_chunks(audio_files)

    def Delete_training_files(self):

        shutil.rmtree(self.Retrain_folder)
        shutil.rmtree(self.Retrain_folder_zip)

    def Retrain_classifiers_for_single_audio(self):

        self.Prepare_audio_chunks(self.wav_file)
        self.Update_embeddings()
        self.Prepare_data_for_training()
        self.Train_novelty_detector()
        accuracy = self.Train_speaker_classifier()
        self.Delete_training_files()

        return accuracy

    def Retrain_classifiers_for_zip_audio(self):

        self.Prepare_audio_chunks_of_zip_file()
        self.Update_embeddings()
        self.Prepare_data_for_training()
        self.Train_novelty_detector()
        accuracy = self.Train_speaker_classifier()
        self.Delete_training_files()

        return accuracy
