# -*- coding: utf-8 -*-
# this only takes in the text file for labelling purpose
# it use the pretrained hubert for processing the audio file into features
# resampled audio is stored in anew wav32k dir and self-supervised hubert feature extraction is stored a pt fle

import sys,os
inp_text= "./tools/output/final_phonetic_transcriptions.txt"
inp_wav_dir= "./tools/denoised/output"
opt_dir="./logs/v1_trial"

from tools import cnhubert
cnhubert.cnhubert_base_path= "./pretrained_models/chinese-hubert-base"

import torch
import traceback,numpy as np
from scipy.io import wavfile
import librosa
now_dir = os.getcwd()
sys.path.append(now_dir)
from tools.my_utils import load_audio,clean_path, get_model_precision


from time import time as ttime
import shutil

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using CUDA")
#elif hasattr(torch, "mps") and torch.backends.mps.is_available():
#    device = torch.device("mps")
#    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

def my_save(fea,path):#####fix issue: torch.save doesn't support chinese path
    dir=os.path.dirname(path)
    name=os.path.basename(path)
    # tmp_path="%s/%s%s.pth"%(dir,ttime(),i_part)
    tmp_path="%s.pth"%(ttime())
    torch.save(fea,tmp_path)
    shutil.move(tmp_path,"%s/%s"%(dir,name))

hubert_dir="%s/4-cnhubert"%(opt_dir)
wav32dir="%s/5-wav32k"%(opt_dir)
os.makedirs(opt_dir,exist_ok=True)
os.makedirs(hubert_dir,exist_ok=True)
os.makedirs(wav32dir,exist_ok=True)

maxx=0.95
alpha=0.5

model=cnhubert.get_model()
precision = get_model_precision(model)
model=model.to(device).to(precision)


nan_fails=[]
def name2go(wav_name,wav_path):
    hubert_path="%s/%s.pt"%(hubert_dir,wav_name)
    if(os.path.exists(hubert_path)):return
    tmp_audio = load_audio(wav_path, 32000)
    tmp_max = np.abs(tmp_audio).max()
    if tmp_max > 2.2:
        print("%s-filtered,%s" % (wav_name, tmp_max))
        return
    tmp_audio32 = (tmp_audio / tmp_max * (maxx * alpha*32768)) + ((1 - alpha)*32768) * tmp_audio
    tmp_audio32b = (tmp_audio / tmp_max * (maxx * alpha*1145.14)) + ((1 - alpha)*1145.14) * tmp_audio
    tmp_audio = librosa.resample(
        tmp_audio32b, orig_sr=32000, target_sr=16000
    )#不是重采样问题
    tensor_wav16 = torch.from_numpy(tmp_audio)
    tensor_wav16 = tensor_wav16.to(device).to(precision)
    ssl=model.model(tensor_wav16.unsqueeze(0))["last_hidden_state"].transpose(1,2).cpu()#torch.Size([1, 768, 215])
    if np.isnan(ssl.detach().numpy()).sum()!= 0:
        nan_fails.append((wav_name,wav_path))
        print("nan filtered:%s"%wav_name)
        return
    wavfile.write(
        "%s/%s"%(wav32dir,wav_name),
        32000,
        tmp_audio32.astype("int16"),
    )
    my_save(ssl,hubert_path)

with open(inp_text,"r",encoding="utf8")as f:
    lines=f.read().strip("\n").split("\n")

for line in lines:
    try:
        # wav_name,text=line.split("\t")
        wav_name, spk_name, language, text = line.split("|")
        wav_name=clean_path(wav_name)
        if (inp_wav_dir != "" and inp_wav_dir != None):
            wav_name = os.path.basename(wav_name)
            wav_path = "%s/%s"%(inp_wav_dir, wav_name)

        else:
            wav_path=wav_name
            wav_name = os.path.basename(wav_name)
        name2go(wav_name,wav_path)
    except:
        print(line,traceback.format_exc())

if(len(nan_fails)>0 and precision=="fp16"):
    model=model.float()
    for wav in nan_fails:
        try:
            name2go(wav[0],wav[1])
        except:
            print(wav_name,traceback.format_exc())
