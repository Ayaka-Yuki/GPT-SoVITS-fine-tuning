# it takes in the ssl features extracted from the previous hubert step
# use pretrained SynthesizerTrn and utilize its ssl_projection and RVQ_quantizer to extract the semantic features

import os

inp_text = "./tools/output/final_phonetic_transcriptions.txt"
opt_dir = "./logs/v1_trial"
pretrained_s2G = "./pretrained_models/s2G488k.pth"
s2config_path = "./configs/s2.json"
import torch
import traceback
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)
import logging, utils
from module.models import SynthesizerTrn
from tools.my_utils import clean_path, get_model_precision
logging.getLogger("numba").setLevel(logging.WARNING)
# from config import pretrained_s2G

if os.path.exists(pretrained_s2G):...
else:raise FileNotFoundError(pretrained_s2G)

hubert_dir = "%s/4-cnhubert" % (opt_dir)
semantic_path = "%s/6-name2semantic.tsv" % (opt_dir)
if os.path.exists(semantic_path) == False:
    os.makedirs(opt_dir, exist_ok=True)

    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using CUDA")
    #elif hasattr(torch, "mps") and torch.backends.mps.is_available():
    #   device = torch.device("mps")
    #    print("Using MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    hps = utils.get_hparams_from_file(s2config_path)
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    precision = get_model_precision(vq_model)
    vq_model = vq_model.to(device).to(precision)
    vq_model.eval()
    # utils.load_checkpoint(utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth"), vq_model, None, True)
    # utils.load_checkpoint(pretrained_s2G, vq_model, None, True)
    print(
        vq_model.load_state_dict(
            torch.load(pretrained_s2G, map_location="cpu",weights_only=False)["weight"],  strict=False
        )
    )

    def name2go(wav_name, lines):
        hubert_path = "%s/%s.pt" % (hubert_dir, wav_name)
        if os.path.exists(hubert_path) == False:
            return
        ssl_content = torch.load(hubert_path, map_location="cpu")
        ssl_content = ssl_content.to(device).to(precision)
        codes = vq_model.extract_latent(ssl_content)
        semantic = " ".join([str(i) for i in codes[0, 0, :].tolist()])
        lines.append("%s\t%s" % (wav_name, semantic))

    with open(inp_text, "r", encoding="utf8") as f:
        lines = f.read().strip("\n").split("\n")

    lines1 = []
    for line in lines:
        # print(line)
        try:
            # wav_name,text=line.split("\t")
            wav_name, spk_name, language, text = line.split("|")
            wav_name=clean_path(wav_name)
            wav_name = os.path.basename(wav_name)
            # name2go(name,lines1)
            name2go(wav_name, lines1)
        except:
            print(line, traceback.format_exc())
    with open(semantic_path, "w", encoding="utf8") as f:
        f.write("\n".join(lines1))
