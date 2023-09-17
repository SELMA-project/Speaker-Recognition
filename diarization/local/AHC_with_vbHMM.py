#!/usr/bin/env python

# Author: Tugtekin Turan
# E-Mail: tugtekin.turan@iais.fraunhofer.de
# Date: 2023-05-19
# Description: Agglomerative hierarchical clustering (AHC) with VBx algorithm

import os

os.environ["KALDI_ROOT"] = "./"

import h5py
import kaldi_io
import argparse
import itertools
import numpy as np
import fastcluster
from typing import Tuple, TextIO
from scipy.special import softmax
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import squareform

# ./local/diarization_lib.py
from diarization_lib import (
    mkdir_p,
    l2_norm,
    cos_similarity,
    twoGMMcalib_lin,
    merge_adjacent_labels,
    read_xvector_timing_dict,
)

# ./local/kaldi_utils.py
from kaldi_utils import read_plda

# ./local/VBx.py
from VBx import VBx


class Diarization:
    def __init__(
        self,
        threshold: float = -0.015,
        lda_dim: int = 128,
        Fa: float = 0.3,
        Fb: float = 17,
        loopP: float = 0.99,
        init_smoothing: float = 5.0,
        output_2nd: bool = False,
    ) -> None:
        # initialize the input arguments
        self.plda_file = read_plda("./diarization/models/plda")
        self.xvec_transform = h5py.File("./diarization/models/transform.h5", "r")
        self.threshold = threshold
        self.lda_dim = lda_dim
        self.Fa = Fa
        self.Fb = Fb
        self.loopP = loopP
        self.init_smoothing = init_smoothing
        self.output_2nd = output_2nd

    def process(
        self, out_rttm_dir: str, xvec_ark_file: str, segments_file: str
    ) -> None:
        """
        Process the input x-vector ark file, write the output RTTM files

        Args:
            out_rttm_dir (str): Directory to store output RTTM files
            xvec_ark_file (str): Kaldi ark file with x-vectors
            segments_file (str): File with timing info of active segments
        """
        # read the x-vectors ark file and group them
        arkit = kaldi_io.read_vec_flt_ark(xvec_ark_file)
        recit = itertools.groupby(arkit, lambda e: e[0].rsplit("_", 1)[0])

        for file_name, segs in recit:
            # unpack the segments and x-vectors
            seg_names, xvecs = zip(*segs)
            x = np.array(xvecs)

            # perform LDA transformation on x-vectors
            x = self.transform_xvectors(x)

            # perform AHC clustering on x-vectors
            labels1st = self.cluster_xvectors(x)

            # perform VB-HMM clustering on output of AHC
            labels1st, labels2nd = self.vbhmm_clustering(x, labels1st)

            # merge adjacent labels and write output rttm files
            segs_dict = read_xvector_timing_dict(segments_file)
            assert np.all(segs_dict[file_name][0] == np.array(seg_names))
            start, end = segs_dict[file_name][1].T
            starts, ends, out_labels = merge_adjacent_labels(
                start, end, labels1st
            )
            mkdir_p(out_rttm_dir)
            with open(
                os.path.join(out_rttm_dir, f"{file_name}.rttm"), "w"
            ) as fp:
                self.write_output(file_name, fp, out_labels, starts, ends)

            if (
                self.output_2nd
                and self.plda_file[2].shape[0] > 1
                and labels2nd is not None
            ):
                starts, ends, out_labels2 = merge_adjacent_labels(
                    start, end, labels2nd
                )
                output_rttm_dir = f"{out_rttm_dir}2nd"
                mkdir_p(output_rttm_dir)
                with open(
                    os.path.join(output_rttm_dir, f"{file_name}.rttm"), "w"
                ) as fp:
                    self.write_output(file_name, fp, out_labels2, starts, ends)

    def transform_xvectors(self, x: np.ndarray) -> np.ndarray:
        """
        Perform LDA transformation on the input x-vectors
        """
        mean1 = np.array(self.xvec_transform["mean1"])
        mean2 = np.array(self.xvec_transform["mean2"])
        lda = np.array(self.xvec_transform["lda"])
        x = l2_norm(
            lda.T.dot((l2_norm(x - mean1)).transpose()).transpose() - mean2
        )
        return x

    def cluster_xvectors(self, x: np.ndarray) -> np.ndarray:
        """
        Cluster x-vectors using Agglomerative Hierarchical Clustering (AHC)
        """
        # pairwise cosine similarities between L2-normalized x-vectors
        scr_mx = cos_similarity(x)

        # calculate the threshold for AHC
        thr, _ = twoGMMcalib_lin(scr_mx.ravel())

        # perform AHC clustering on the x-vectors
        scr_mx = squareform(-scr_mx, checks=False)
        lin_mat = fastcluster.linkage(
            scr_mx, method="average", preserve_input=False
        )
        del scr_mx
        adjust = abs(lin_mat[:, 2].min())
        lin_mat[:, 2] += adjust
        labels1st = (
            fcluster(
                lin_mat, -(thr + self.threshold) + adjust, criterion="distance"
            )
            - 1
        )
        return labels1st

    def vbhmm_clustering(
        self, x: np.ndarray, labels1st: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform VB-HMM clustering on the output of AHC clustering

        Args:
            x (np.ndarray): Matrix of L2-normalized x-vectors
            labels1st (np.ndarray): Array of labels obtained from AHC clustering
        """
        # initialize the occupation probability distribution
        qinit = np.zeros((len(labels1st), np.max(labels1st) + 1))
        qinit[range(len(labels1st)), labels1st] = 1.0
        qinit = softmax(qinit * self.init_smoothing, axis=1)

        # apply PLDA transformation to x-vectors
        fea = (x - self.plda_file[0]).dot(self.plda_file[1].T)[
            :, : self.lda_dim
        ]

        # perform VBx clustering
        q, sp, L = VBx(
            fea,
            self.plda_file[2][: self.lda_dim],
            pi=qinit.shape[1],
            gamma=qinit,
            maxIters=40,
            epsilon=1e-6,
            loopProb=self.loopP,
            Fa=self.Fa,
            Fb=self.Fb,
        )

        # get the updated labels after VB-HMM clustering
        labels1st = np.argsort(-q, axis=1)[:, 0]
        labels2nd = None
        if q.shape[1] > 1:
            # also save the second best labels after the clustering
            labels2nd = np.argsort(-q, axis=1)[:, 1]

        return labels1st, labels2nd

    def write_output(
        self,
        file_name: str,
        fp: TextIO,
        out_labels: np.ndarray,
        starts: np.ndarray,
        ends: np.ndarray,
    ) -> None:
        """
        Writes the output RTTM file
        """
        for label, seg_start, seg_end in zip(out_labels, starts, ends):
            fp.write(
                f"SPEAKER {file_name} 1 {seg_start:03f} {seg_end - seg_start:03f} "
                f"<NA> <NA> {label + 1} <NA> <NA>{os.linesep}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-rttm-dir",
        required=True,
        type=str,
        help="Directory to store output rttm files",
    )
    parser.add_argument(
        "--xvec-ark-file",
        required=True,
        type=str,
        help="Kaldi ark file with x-vectors",
    )
    parser.add_argument(
        "--segments-file",
        required=True,
        type=str,
        help="File with x-vector timing info",
    )
    args = parser.parse_args()

    # initialize the x-vector diarization object
    xvec_diarization = Diarization()

    # start diarization process and write RTTM file
    xvec_diarization.process(
        args.out_rttm_dir, args.xvec_ark_file, args.segments_file
    )

    print(f"Diarization is completed! Check files under {args.out_rttm_dir}\n")
