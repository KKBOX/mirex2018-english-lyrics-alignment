# MIREX2018 English Lyrics Alignment

This repository shows our submissions (CW2 and CW3) for subtask 2 of lyrics alignment of MIREX2018

## Installation and Requirements

* HTK and ffmpeg
  * Set the $PATH enviroment variable if needed
* Python 3.6 and libraries in requirements.txt
* Download models and example data from [here](https://drive.google.com/drive/folders/1EKdsSoiFI0Zg1KMe7nnpApBnGfG_zrKB)
  * Example data came from the [Mauch dataset](https://www.music-ir.org/mirex/wiki/2018:Automatic_Lyrics-to-Audio_Alignment#Mauch.27s_Dataset) provided in MIREX 2018

## Usage

* Calling format: `python3 go.py %input_audio %input_txt %output_txt`
  * Example: `python3 go.py example_data/Muse.GuidingLight.mp3 example_data/Muse.GuidingLight.txt output.txt`
* To switch between CW2 and CW3, add `--model_dir model_CW2` or `--model_dir model_CW3` respectively. The default is model\_CW3
* The running time is about 2.33 min for Muse.GuidingLight.mp3 on a machine with 2.50GHz CPU
