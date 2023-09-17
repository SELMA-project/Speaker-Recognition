#!/usr/bin/env python

# Author: Tugtekin Turan
# E-Mail: tugtekin.turan@iais.fraunhofer.de
# Date: 2023-05-19
# Description: Tests for the audio segmentation pipeline

"""
Usage:
    Install "pytest" first, then execute this script from the main directory.
    It will directly use the sample ".wav" file under the test/ folder.

Example:
    $ pytest test/test_diarization.py
"""

import os
import sys
import pytest
import subprocess
import shutil
import numpy as np

# append directories to the PYTHONPATH
sys.path += ["./local", "./utils"]

from create_vad_segments import VAD
from extract_embeddings import EmbeddingExtraction
from AHC_with_vbHMM import Diarization


@pytest.fixture
def input_file():
    return "./test/tagesschau02092019.wav"


@pytest.fixture
def input_ref():
    return "./test/tagesschau02092019.rttm"


@pytest.fixture
def vad():
    return VAD()


@pytest.fixture
def embedding_extraction():
    return EmbeddingExtraction()


@pytest.fixture
def diarization():
    return Diarization()


def test_pipeline(
    input_file, input_ref, vad, embedding_extraction, diarization
):
    # create temporary directory
    temp_dir = "./test_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    # define the output file paths
    label_dir = os.path.join(temp_dir, "vad_files")
    out_lab_file = os.path.join(label_dir, "tagesschau02092019.lab")
    out_ark_file = os.path.join(temp_dir, "tagesschau02092019.ark")
    out_seg_file = os.path.join(temp_dir, "tagesschau02092019.seg")
    out_rttm_file = os.path.join(temp_dir, "tagesschau02092019.rttm")

    # perform voice activity detection
    vad.extract(input_file, temp_dir)

    # check that the VAD label file was created and the last segment is correct
    assert os.path.exists(out_lab_file)
    expected_last_lab = np.array([866.52, 880.47])
    assert np.array_equal(np.loadtxt(out_lab_file)[-1], expected_last_lab)

    # perform embedding extraction
    embeddings = embedding_extraction.compute_embeddings(input_file, label_dir)
    embedding_extraction.write_embeddings_to_ark(out_ark_file, embeddings)
    embedding_extraction.write_segments_to_txt(
        out_seg_file, embeddings, input_file
    )

    # check that the embedding and segment files were created
    # also check the last segment is correct
    assert os.path.exists(out_ark_file)
    assert os.path.exists(out_seg_file)
    expected_last_seg = np.array([879.24, 880.47])
    assert np.array_equal(
        np.loadtxt(out_seg_file, dtype=str)[-1][-2:].astype(float),
        expected_last_seg,
    )

    # perform speaker diarization
    diarization.process(temp_dir, out_ark_file, out_seg_file)

    # check that the RTTM file was created and the diarization error is correct
    assert os.path.exists(out_rttm_file)
    cmd = f"./test/md-eval.pl -1 -c 0.25 -r {input_ref} -s {out_rttm_file}"
    result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True)
    output = result.stdout.decode("utf-8")
    assert "DIARIZATION ERROR = 2.69" in output

    # clean up the temporary directory
    shutil.rmtree(temp_dir)
