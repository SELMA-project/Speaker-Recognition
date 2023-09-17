#!/usr/bin/env python

# Author: Tugtekin Turan
# E-Mail: tugtekin.turan@iais.fraunhofer.de
# Date: 2023-05-19
# Description: Preparation of voice activity detection (VAD) segments

import os
import json
import torch
import shutil
import logging
import argparse
import jsonlines
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# set the log severity level to "FATAL"
logging.disable(logging.FATAL)

from nemo.collections.asr.parts.utils.vad_utils import (
    generate_vad_frame_pred,
    generate_overlap_vad_seq,
    generate_vad_segment_table,
    init_vad_model,
    prepare_manifest,
)
from nemo.collections.asr.parts.utils.speaker_utils import write_rttm2manifest

# use CPU instead of GPU
device = torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# disable gradient calculation (equivalent to "torch.no_grad()")
torch.set_grad_enabled(False)

# load VAD model
config = OmegaConf.load("./diarization/utils/vad_config.yaml")
vad_model = init_vad_model(config.vad.model_path)
vad_model = vad_model.to(device)
vad_model.eval()


class VAD:
    def __init__(
        self,
        config: DictConfig = config,
        vad_model: torch.nn.Module = vad_model,
    ) -> None:
        self.vad_model = vad_model
        self.config = config

    def generate_meta(self) -> None:
        # set up the file paths and manifest dictionary for the input file
        self.base_name = os.path.splitext(os.path.basename(self.input_file))[0]
        self.manifest_file = os.path.join(
            self.output_folder, self.base_name + ".json"
        )

        meta = {
            "audio_filepath": self.input_file,
            "num_speakers": None,
            "duration": None,
            "label": "infer",
            "offset": 0,
            "text": "-",
        }

        # write the manifest file to disk
        with open(self.manifest_file, "w") as out:
            json.dump(meta, out)
            out.write("\n")

        # store the manifest information in a dictionary for future use
        self.key_meta_map = {}
        self.key_meta_map[self.base_name] = {"audio_filepath": self.input_file}

    def prepare_manifest_vad_input(self) -> None:
        # prepare the manifest file for VAD input
        manifest_config = {
            "input": self.manifest_file,
            "window_length_in_sec": self.config.vad.parameters.window_length_in_sec,
            "split_duration": self.config.prepare_manifest.split_duration,
            "num_workers": self.config.num_workers,
        }
        manifest_vad_input = prepare_manifest(manifest_config)

        # move the temporary file to the output directory
        shutil.move(manifest_vad_input, self.config.prepared_manifest_vad_input)

    def setup_test_data(self) -> None:
        # sets up the test data for VAD by creating a PyTorch Dataset object
        self.vad_model.setup_test_data(
            test_data_config={
                "vad_stream": True,
                "sample_rate": 16000,
                "manifest_filepath": self.config.prepared_manifest_vad_input,
                "labels": ["infer"],
                "num_workers": self.config.num_workers,
                "shuffle": False,
                "window_length_in_sec": self.config.vad.parameters.window_length_in_sec,
                "shift_length_in_sec": self.config.vad.parameters.shift_length_in_sec,
                "trim_silence": False,
                "normalize_audio": self.config.vad.parameters.normalize_audio,
            }
        )

    def generate_vad_frame(self) -> None:
        # create "vad_files" directory and save predictions for each frame in JSON
        self.config.frame_out_dir = os.path.join(
            self.output_folder, "vad_files"
        )
        if os.path.exists(self.config.frame_out_dir):
            shutil.rmtree(self.config.frame_out_dir, ignore_errors=True)
        os.makedirs(self.config.frame_out_dir)

        self.pred_dir = generate_vad_frame_pred(
            vad_model=self.vad_model,
            window_length_in_sec=self.config.vad.parameters.window_length_in_sec,
            shift_length_in_sec=self.config.vad.parameters.shift_length_in_sec,
            manifest_vad_input=self.config.prepared_manifest_vad_input,
            out_dir=self.config.frame_out_dir,
        )

    def generate_overlap_seq(self) -> None:
        # generate overlapped and smoothed VAD sequence using frame-level predictions
        self.smoothing_pred_dir = generate_overlap_vad_seq(
            frame_pred_dir=self.pred_dir,
            smoothing_method=self.config.vad.parameters.smoothing,
            overlap=self.config.vad.parameters.overlap,
            window_length_in_sec=self.config.vad.parameters.window_length_in_sec,
            shift_length_in_sec=self.config.vad.parameters.shift_length_in_sec,
            num_workers=self.config.num_workers,
            out_dir=self.config.smoothing_out_dir,
        )

    def generate_vad_segment_table(self) -> None:
        # create final segment table by post-processing the output
        self.table_out_dir = generate_vad_segment_table(
            vad_pred_dir=self.smoothing_pred_dir,
            postprocessing_params=self.config.vad.parameters.postprocessing,
            frame_length_in_sec=self.config.vad.parameters.shift_length_in_sec,
            num_workers=self.config.num_workers,
            out_dir=self.config.table_out_dir,
        )

    def write_sad(self) -> None:
        # create RTTM file for each audio file in the input manifest
        for i in self.key_meta_map:
            self.key_meta_map[i]["rttm_filepath"] = os.path.join(
                self.table_out_dir, i + ".txt"
            )
        os.remove(os.path.join(os.getcwd(), "./temp_results/manifest_vad_input.json"))
        manifest_path = os.path.join(self.config.frame_out_dir, "vad_out.json")
        write_rttm2manifest(self.key_meta_map, manifest_path)

        # create dataframe containing start and end times of VAD segments
        df = pd.DataFrame(columns=["start", "end"])
        with jsonlines.open(manifest_path) as f:
            for line in f.iter():
                row = [line["offset"], line["offset"] + line["duration"]]
                df.loc[len(df)] = row

        # save VAD segments to a ".lab" file
        sad_file = os.path.join(
            self.config.frame_out_dir, self.base_name + ".lab"
        )
        np.savetxt(sad_file, df.values, fmt="%f")

    def extract(self, input_file, output_folder) -> None:
        # main function that orchestrates the VAD extraction process
        self.input_file = input_file
        self.output_folder = output_folder
        self.generate_meta()
        self.prepare_manifest_vad_input()
        self.setup_test_data()
        self.generate_vad_frame()
        self.generate_overlap_seq()
        self.generate_vad_segment_table()
        self.write_sad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract speaker activity detection segments"
    )
    parser.add_argument(
        "-i",
        "--input-file",
        required=True,
        type=str,
        help="Path to the input .wav file",
    )
    parser.add_argument(
        "-o",
        "--output-folder",
        required=True,
        type=str,
        help="Path to the ouput experiment folder",
    )
    args = parser.parse_args()

    # create a VAD instance and run the extraction
    segments = VAD()
    segments.extract(args.input_file, args.output_folder)

    print(f"VAD processing of the {args.input_file} is completed!\n")
