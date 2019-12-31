# MIREX2018 English Lyrics Alignment

This repository shows our submissions (CW2 and CW3) for subtask 2 of lyrics alignment of MIREX2018. HTK and Tensorflow are used in our submission. More details can be found in [our abstract](https://www.music-ir.org/mirex/abstracts/2018/CW2.pdf).

## Requirements

* FFmpeg
* [Git LFS](https://help.github.com/en/github/managing-large-files/installing-git-large-file-storage)
* HTK 3.4.1
* Python 3.6

## Usage

* Calling format: `python3 go.py %input_audio %input_txt %output_txt`
  * Example: `python3 go.py example_data/Muse.GuidingLight.mp3 example_data/Muse.GuidingLight.txt output.txt`
* To switch between CW2 and CW3, add `--model_dir model_CW2` or `--model_dir model_CW3` respectively. The default is model\_CW3.
  * To use your own model, put the files of configuration (for HCopy), dictionary, list of models, and macro in a directory. Those files should be named mfcc39.edaz.cfg, dic.dic, model_list.model, and macro.final, respectively.
* The running time is about 2.33 min for Muse.GuidingLight.mp3 on a machine with 2.50GHz CPU.

## Notes

* Example data came from the [Mauch dataset](https://www.music-ir.org/mirex/wiki/2018:Automatic_Lyrics-to-Audio_Alignment#Mauch.27s_Dataset) provided in MIREX 2018.
