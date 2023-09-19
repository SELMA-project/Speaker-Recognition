# Introduction

VBx-HMM clustering uses a statistical model to represent each speaker and a sequence model to represent the order of speakers. It then uses variational Bayes (VB) to estimate these models from the data. This method begins with extracting x-vectors from the audio data. These x-vectors are then processed using an HMM, where each state corresponds to a different speaker and transitions between states represent speaker changes. The model uses a single Gaussian component to represent each speaker's distribution, simplifying the process compared to previous methods. The inference of the most likely sequence of speakers is performed using VB, which approximates the complex posterior distribution of the model.

# Getting Started
Install the required packages,

```bash
$ pip install -r requirements.txt
```

To test it under an isolated workspace, create a Conda environment from the YAML file,

```bash
$ conda env create -n <new_env_name> -f conda_env.yaml
```

# Quick Usage

```python
>>> from src.diarization import Pipeline
>>> diarizer = Pipeline.init_from_wav("./test/tagesschau02092019.wav")
>>> rttm_path = diarizer.write_to_RTTM("./output_folder")
```

This will use input `tagesschau02092019.wav` file (sampled at 16 kHz) and creates RTTM-formatted output with other experimental files into the `output_folder/`. Please check the [example notebook](./example_notebook.ipynb) for this file.

# Output Files

|                                  | Description                                                    |
| :------------------------------: | -------------------------------------------------------------- |
| `./output_folder/test_file.ark`  | Speaker embeddings for each segment stored in Kaldi ark format |
| `./output_folder/test_file.seg`  | Segment start/end times for each segment in a plain text file  |
| `./output_folder/test_file.rttm` | Speaker diarization results for the input file in RTTM format  |

# Performance Evaluations

If you provide the ground-truth segments, it is possible to calculate the objective results using the `md-eval.pl` script. Please follow the following way for `test/tagesschau02092019.rttm` file,

```bash
$ ./test/md-eval.pl -1 -c 0.25 -r test/tagesschau02092019.rttm -s output_folder/tagesschau02092019.rttm
```

where `-s` refers system output and `-r` contains reference metadata respectively.

This calculation will return a report whose important parts are as follows,

```
---------------------------------------------
SCORED SPEAKER TIME =    795.18 secs (100.00 percent of scored speech)
MISSED SPEAKER TIME =      9.79 secs ( 1.23 percent of scored speaker time)
FALARM SPEAKER TIME =      0.60 secs ( 0.08 percent of scored speaker time)
 SPEAKER ERROR TIME =     11.03 secs ( 1.39 percent of scored speaker time)
SPEAKER ERROR WORDS =      0         (100.00 percent of scored speaker words)
---------------------------------------------
 OVERALL SPEAKER DIARIZATION ERROR = 2.69 percent of scored speaker time  `(ALL)
---------------------------------------------
 Speaker type confusion matrix -- speaker weighted
  REF\SYS (count)      unknown               MISS
unknown                  27 /  93.1%          2 /   6.9%
  FALSE ALARM             0 /   0.0%
---------------------------------------------
```
Here, false alarm and missed speaker times are related to the performance of voice activity detection whereas speaker error time is the result of the clustering performance. The overall diarization error rate (DER) is calculated as `2.69%` for this example and all the `27` of a total of `29` speakers were correctly identified.
