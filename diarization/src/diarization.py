import sys
import shutil
import logging
from pathlib import Path

# set the log severity level to "FATAL"
logging.disable(logging.FATAL)

# append directories to the PYTHONPATH
sys.path += ["./diarization/local", "./diarization/utils"]

from create_vad_segments import VAD
from AHC_with_vbHMM import Diarization
from extract_embeddings import EmbeddingExtraction


class Pipeline:
    def __init__(self, input_file):
        self.input_file = Path(input_file)

    def diarize(self, output_dir):
        """Orchestration method that performs diarization on the input audio file"""
        self.create_output_directory(output_dir)
        segments = self.extract_segments(output_dir)
        embeddings = self.compute_embeddings(segments, output_dir)
        ark_file, seg_file = self.write_output_files(embeddings, output_dir)
        rttm_path = self.run_diarization(ark_file, seg_file, output_dir)
        return rttm_path

    def extract_segments(self, output_dir):
        """Extracts segments from the input audio file using VAD module"""
        sys.stdout = sys.__stdout__
        segments = VAD()
        segments.extract(str(self.input_file), str(output_dir))
        return segments

    def compute_embeddings(self, segments, output_dir):
        """Computes embeddings from the VAD segments using an embedding extraction model"""
        extractor = EmbeddingExtraction()
        label_file = output_dir.joinpath("vad_files")
        embeddings = extractor.compute_embeddings(str(self.input_file), str(label_file))
        return embeddings

    def write_output_files(self, embeddings, output_dir):
        """Write segment files and embedding features in formats suitable for diarization"""
        file_name = self.input_file.stem
        ark_file = output_dir.joinpath(f"{file_name}.ark")
        seg_file = output_dir.joinpath(f"{file_name}.seg")
        EmbeddingExtraction.write_embeddings_to_ark(ark_file, embeddings)
        EmbeddingExtraction.write_segments_to_txt(seg_file, embeddings, str(self.input_file))
        return ark_file, seg_file

    def run_diarization(self, ark_file, seg_file, output_dir):
        """Runs the diarization algorithm over the embeddings and VAD segments"""
        xvec_diarization = Diarization()
        xvec_diarization.process(str(output_dir), str(ark_file), str(seg_file))
        file_name = self.input_file.stem
        rttm_path = output_dir.joinpath(f"{file_name}.rttm")
        return rttm_path

    def write_to_RTTM(self, output_dir):
        """Writes the results of diarization to an RTTM file"""
        output_dir = Path(output_dir)
        rttm_path = self.diarize(output_dir)
        print(f"Diarization completed. RTTM file generated at {rttm_path}")
        return str(rttm_path)

    def create_output_directory(self, output_dir):
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

    @classmethod
    def init_from_wav(cls, input_file):
        return cls(input_file)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run speaker diarization")
    parser.add_argument("input_file", help="path to input WAV file")
    parser.add_argument("output_dir", help="path to output directory")
    args = parser.parse_args()

    diarizer = Pipeline.init_from_wav(args.input_file)
    rttm_file = diarizer.write_to_RTTM(args.output_dir)
