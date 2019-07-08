import os
import re
import sys
import random
import string
import argparse
import subprocess

import numpy as np
import scipy.io.wavfile


def loadFile(filePath):
    with open(filePath, 'r') as fin:
        return fin.read().splitlines()


parser = argparse.ArgumentParser()
parser.add_argument('input_audio', help='')
parser.add_argument('input_txt', help='')
parser.add_argument('output', help='')
args = parser.parse_args()

if sys.version_info.major != 3:
    raise Exception('This program uses Python3.')

add_sil_at_line_end = True
vnv_model = 'vnv_model'
do_vnv = True
vnv_hop = 160

song_name = args.input_audio.split('/')[-1].split('.')[0]
tmp_dir = 'tmp_{}_{}'.format(song_name, ''.join(random.choice(
    string.ascii_uppercase + string.digits) for _ in range(5))
)
if os.path.isdir(tmp_dir):
    raise Exception('Tmp directory already exist!')
else:
    os.makedirs(tmp_dir)

tmp_grm_file = os.path.join(tmp_dir, 'tmp.grm')
tmp_net_file = os.path.join(tmp_dir, 'tmp.net')
tmp_wav_file = os.path.join(tmp_dir, 'tmp.wav')
tmp_out_file = os.path.join(tmp_dir, 'tmp.mlf')
tmp_dic_file = os.path.join(tmp_dir, 'tmp.dic')
# ---
tmp_vnv_txt = os.path.join(tmp_dir, 'tmp.vnv.txt')
tmp_vnv_wav = os.path.join(tmp_dir, 'tmp.vnv.wav')

cfg_file = 'htk_dict_and_data/mfcc39.edaz.cfg'
dic_file = 'htk_dict_and_data/eng.biphone.dic'
model_file = 'htk_dict_and_data/ly_west_7300.model'
macro_file = 'htk_macro/macro.ly_west_seg_7300.bi.24.final'

# Read dict
try:
    dic_cnt = loadFile(dic_file)
    eng_dic = {d.split()[0]: d.split()[1:] for d in dic_cnt}
except Exception as e:
    raise Exception('Read dictionary error')

# Read model
try:
    cnt = loadFile(model_file)
    while cnt[-1] == '':
        cnt = cnt[:-1]
    all_model = [c.split()[0] if len(c.split()) <= 1 else c.split()[1] for c in cnt]
except Exception as e:
    raise Exception('Read model error')

# Call ffmpeg to get 16 kHz mono file
exit_code = subprocess.call(['ffmpeg', '-i', args.input_audio, '-ar', '16000', '-ac', '1', '-y', tmp_wav_file])
if exit_code != 0:
    raise Exception('ffmpeg error')

if do_vnv:
    subprocess.call([
        'python3', 'goVnv.py', vnv_model, '--overwrite_output', '-i', tmp_wav_file, '-o', tmp_vnv_txt
    ])
    # Read files
    vnv_result = np.loadtxt(tmp_vnv_txt)
    fs, yOri = scipy.io.wavfile.read(tmp_wav_file)
    # Generate vnv data
    sample_vnv = np.zeros_like(yOri)
    for ft, vnv, _, _ in vnv_result:
        if vnv == 1:
            s1 = max(0, int(ft * fs - vnv_hop))
            s2 = min(int(ft * fs + vnv_hop), len(sample_vnv))
            sample_vnv[s1:s2] = 1
    # make new wav file
    yNew = yOri[sample_vnv == 1]
    scipy.io.wavfile.write(tmp_vnv_wav, fs, yNew)
    tmp_wav_file = tmp_vnv_wav
    # make true-alignment lookup table
    sample_idx_ori = np.arange(len(sample_vnv))
    sample_idx_new = sample_idx_ori[sample_vnv == 1]

# Read input text
cnt = loadFile(args.input_txt)
word_list_recog = ['sil']  # Beginning sil
word_list_output = ['sil']
tmp_dic_list = ['sil']
for line_no, line_txt in enumerate(cnt):
    for word in line_txt.split():
        word_clean = re.sub(r'[0-9\.,?!]', '', word).lower()
        word_list_output.append(word)
        if word_clean in eng_dic or '\\\\'+word_clean in eng_dic:
            word_models = eng_dic[word_clean] if word_clean in eng_dic else eng_dic['\\\\'+word_clean]
            all_model_seen = True
            for wm in word_models:
                if wm not in all_model:
                    all_model_seen = False
            if all_model_seen:
                word_list_recog.append(word_clean)
                tmp_dic_list.append(word_clean)
            else:
                word_list_recog.append('sil')
                tmp_dic_list.append('sil')
        else:
            word_list_recog.append('sil')
            tmp_dic_list.append('sil')
    # Add sil at line end
    if add_sil_at_line_end and line_no < len(cnt) - 1:
        word_list_output.append('')
        word_list_recog.append('sil')
word_list_output.append('sil')  # Ending sil
word_list_recog.append('sil')

# Output temp dict
tmp_dic_list = sorted(list(set(tmp_dic_list)))
with open(tmp_dic_file, 'w') as fout:
    for word in tmp_dic_list:
        model = eng_dic[word] if word in eng_dic else eng_dic['\\\\'+word]
        fout.write('{} {}\n'.format(word, ' '.join(model)))
dic_file = tmp_dic_file

# Write word_list_recog to tmp grm/net file
with open(tmp_grm_file, 'w') as fout:
    fout.write('(')
    for w in word_list_recog:
        fout.write('(' + w + ')')
    fout.write(')')
exit_code = subprocess.call(['HParse', tmp_grm_file, tmp_net_file])
if exit_code != 0:
    raise Exception('HParse error')

# Call HVite
exit_code = subprocess.call(['HVite', '-H', macro_file, '-i', tmp_out_file, '-w',
                             tmp_net_file, '-C', cfg_file, dic_file, model_file, tmp_wav_file])
if exit_code != 0:
    raise Exception('HVite error')

# Read mlf
try:
    cnt = loadFile(tmp_out_file)[3:][:-2]  # Eliminate fixed sil in the beginning and end
    word_align = [c.split(' ')[:-1] for c in cnt]
except Exception as e:
    raise('Reading tmp_out_file error')

if do_vnv:
    for wa in word_align:
        t1 = float(wa[0]) / 1e7
        t2 = float(wa[1]) / 1e7
        s1 = int(t1 * 16000)
        s2 = int(t2 * 16000)
        s1_ori = sample_idx_new[s1]
        s2_ori = sample_idx_new[s2]
        t1_ori = s1_ori / 16000
        t2_ori = s2_ori / 16000
        wa[0] = t1_ori * 1e7
        wa[1] = t2_ori * 1e7

# write to output file
with open(args.output, 'w') as fout:
    for i, m in enumerate(word_align):
        t1 = float(m[0]) / 1e7
        t2 = float(m[1]) / 1e7
        word = word_list_output[i+1]
        if word != '':
            fout.write('{}\t{}\t{}\n'.format(t1, t2, word))
