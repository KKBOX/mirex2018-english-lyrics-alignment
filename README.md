# MIREX2018 English Lyrics Alignment

This repository shows our submissions (CW2 and CW3) for subtask 2 of lyrics alignment of MIREX2018

## Installation and Requirements

* HTK and ffmpeg
  * Set the $PATH enviroment variable if needed
* Python 3.6 and libraries in requirements.txt
* Download models and example data from [here](https://drive.google.com/drive/folders/1EKdsSoiFI0Zg1KMe7nnpApBnGfG_zrKB)
  * Example data came from the Mauch dataset

## Usage

- For subtask 2, songs with with instrumental accompaniment
- Calling format: `python3 go.py %input_audio %input_txt %output_txt`
- Tool/library requirements:
  - Python3.6
  - tensorflow==1.3.0
  - numpy==1.15.0
  - scipy==1.1.0
  - ffmpeg
  - HTK
  - The path of HTK and ffmpeg should be in the $PATH enviroment variable
  - see requirements.txt for details of python libraries, if needed
- Took about 3.75 min for Muse.GuidingLight.mp3 on a machine with 2.30GHz CPU
