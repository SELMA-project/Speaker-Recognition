#!/usr/bin/env python

# Author: Tugtekin Turan
# E-Mail: tugtekin.turan@iais.fraunhofer.de
# Date: 2023-02-19
# Description: Speaker embedding extractions over x-vectors

import os

os.environ["KALDI_ROOT"] = "./"

import torch
import kaldi_io
import argparse
import onnxruntime
import numpy as np
import soundfile as sf
from typing import Dict, Tuple

import features  # ./local/features.py

# use CPU instead of GPU
device = torch.device("cpu")
torch.backends.cudnn.enabled = False
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# disable gradient calculation (equivalent to "torch.no_grad()")
torch.set_grad_enabled(False)

# set the log severity level to "FATAL"
onnxruntime.set_default_logger_severity(3)

# load speaker embeddings model
model_path = "./diarization/models/embedding.onnx"
model = onnxruntime.InferenceSession(model_path)

# for reproducibility
np.random.seed(3)


class EmbeddingExtraction:
    def __init__(
        self,
        model=model,
        ndim: int = 64,
        embed_dim: int = 256,
        seg_len: int = 144,
        seg_jump: int = 24,
    ) -> None:
        self.model = model
        self.input_name = self.model.get_inputs()[0].name
        self.label_name = self.model.get_outputs()[0].name
        self.ndim = ndim
        self.seg_len = seg_len
        self.seg_jump = seg_jump
        self.embed_dim = embed_dim
        self.lc = 150
        self.rc = 149
        self.sr = 16000
        self.winlen = 400
        self.noverlap = 240
        self.window = features.povey_window(self.winlen)
        self.fbank_mx = features.mel_fbank_mx(
            self.winlen,
            self.sr,
            NUMCHANS=self.ndim,
            LOFREQ=20.0,
            HIFREQ=7600,
            htk_bug=False,
        )

    @staticmethod
    def write_embeddings_to_ark(
        out_ark_file: str,
        embeddings: Dict[str, Tuple[float, float, np.ndarray]],
    ) -> None:
        # write the embeddings to a Kaldi archive file
        with open(out_ark_file, "wb") as ark_file:
            for key in sorted(embeddings.keys()):
                _, _, embedding = embeddings[key]
                kaldi_io.write_vec_flt(ark_file, embedding, key=key)

    @staticmethod
    def write_segments_to_txt(
        out_seg_file: str,
        embeddings: Dict[str, Tuple[float, float, np.ndarray]],
        filename: str,
    ) -> None:
        # write the segments and corresponding start and end times to a text file
        base = os.path.splitext(os.path.basename(filename))[0]
        with open(out_seg_file, "w") as f:
            for key, (start_time, end_time, _) in embeddings.items():
                f.write(f"{key} {base} {start_time} {end_time}\n")

    def get_embedding(self, fea: np.ndarray) -> np.ndarray:
        # transpose and add an extra dimension to the feature matrix
        input_data = fea.astype(np.float32).transpose()[np.newaxis, :, :]
        # run the model on the input data to get the embedding vector
        output = self.model.run(
            [self.label_name], {self.input_name: input_data}
        )[0]
        # return the embedding vector of shape (embedding_dim,)
        return output.squeeze()

    def compute_embeddings(
        self,
        audio_path: str,
        lab_dir: str,
    ) -> Dict[str, Tuple[float, float, np.ndarray]]:
        # read audio file
        signal = sf.read(audio_path)
        signal = signal[0].astype(np.float32)
        signal = features.add_dither((signal * 2**15).astype(int))

        # load VAD labels
        fn = os.path.splitext(os.path.basename(audio_path))[0]
        labs = (
            np.loadtxt(os.path.join(lab_dir, fn + ".lab"), usecols=(0, 1))
            * self.sr
        )
        labs = np.atleast_2d(labs.astype(int))

        embeddings = {}
        for segnum in range(len(labs)):
            seg = signal[labs[segnum, 0] : labs[segnum, 1]]

            # process segments only if longer than 0.01 sec
            if seg.shape[0] > 0.01 * self.sr:
                # mirror noverlap // 2 initial and final samples
                seg = np.r_[
                    seg[self.noverlap // 2 - 1 :: -1],
                    seg,
                    seg[-1 : -self.winlen // 2 - 1 : -1],
                ]
                fea = features.fbank_htk(
                    seg,
                    self.window,
                    self.noverlap,
                    self.fbank_mx,
                    USEPOWER=True,
                    ZMEANSOURCE=True,
                )
                fea = features.cmvn_floating_kaldi(
                    fea, self.lc, self.rc, norm_vars=False
                ).astype(np.float32)

                slen = len(fea)
                start = -self.seg_jump

                # iterate over segments in the input speech signal
                for start in range(0, slen - self.seg_len, self.seg_jump):
                    data = fea[start : start + self.seg_len]
                    # compute the embedding for the segment
                    xvector = self.get_embedding(data)
                    if not np.isnan(xvector).any():
                        key = f"{fn}_{segnum:04}-{start:08}-{(start + self.seg_len):08}"
                        seg_start = round(
                            labs[segnum, 0] / float(self.sr) + start / 100.0, 3
                        )
                        seg_end = round(
                            labs[segnum, 0] / float(self.sr)
                            + start / 100.0
                            + self.seg_len / 100.0,
                            3,
                        )
                        embeddings[key] = (seg_start, seg_end, xvector)

                # add any remaining part of the signal as a segment
                if slen - start - self.seg_jump >= 10:
                    data = fea[start + self.seg_jump : slen]
                    # compute the embedding for the segment
                    xvector = self.get_embedding(data)
                    if not np.isnan(xvector).any():
                        key = f"{fn}_{segnum:04}-{(start + self.seg_jump):08}-{slen:08}"
                        seg_start = round(
                            labs[segnum, 0] / float(self.sr)
                            + (start + self.seg_jump) / 100.0,
                            3,
                        )
                        seg_end = round(labs[segnum, 1] / float(self.sr), 3)
                        embeddings[key] = (seg_start, seg_end, xvector)

        return embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract speaker embeddings over overlapping segments"
    )
    parser.add_argument(
        "--in-wav-file",
        required=True,
        type=str,
        help="Local path of input .wav file",
    )
    parser.add_argument(
        "--in-lab-dir",
        required=True,
        type=str,
        help="Input directory containing VAD labels",
    )
    parser.add_argument(
        "--out-ark-file", required=True, type=str, help="Output embedding file"
    )
    parser.add_argument(
        "--out-seg-file", required=True, type=str, help="Output segments file"
    )
    args = parser.parse_args()

    # compute embeddings for each segment
    extractor = EmbeddingExtraction()
    embeddings = extractor.compute_embeddings(args.in_wav_file, args.in_lab_dir)

    # write output files
    EmbeddingExtraction.write_embeddings_to_ark(args.out_ark_file, embeddings)
    EmbeddingExtraction.write_segments_to_txt(
        args.out_seg_file, embeddings, args.in_wav_file
    )

    print(f"X-Vector extraction of {args.in_wav_file} file is completed!\n")
