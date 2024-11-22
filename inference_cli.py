import os, re, sys, json, argparse
import soundfile as sf

import torch
import numpy as np
import librosa

from tools.i18n.i18n import I18nAuto
i18n = I18nAuto()

from transformers import AutoTokenizer, AutoModelForMaskedLM
from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from langdetect import detect
import nltk
import LangSegment
import chardet
from tools.my_utils import get_model_precision, load_audio
from module.mel_processing import spectrogram_torch

if torch.cuda.is_available():
    device = "cuda"
#elif torch.backends.mps.is_available():
#    device = "mps"
else:
    device = "cpu"

splits = ["。", "？", "！", ".", "?", "!", "~", ":", "：", "—", "…"]

class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")
        
# load bert from bert_path
bert_path = "./pretrained_models/chinese-roberta-wwm-ext-large"
tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
bert_precision = get_model_precision(bert_model)
bert_model = bert_model.to(device).to(bert_precision)

def change_sovits_weights(sovits_path):
    global vq_model, hps, vq_precision
    dict_s2 = torch.load(sovits_path, map_location="cpu",weights_only=False)
    hps = dict_s2["config"]
    if dict_s2['weight']['enc_p.text_embedding.weight'].shape[0] == 322:
        hps.model.version = "v1"
    else:
        hps.model.version = "v2"
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    vq_precision = get_model_precision(vq_model)
    vq_model = vq_model.to(device).to(vq_precision)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))

def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    maxx=audio.abs().max()
    if(maxx>1):audio/=min(2,maxx)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec
    
def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu",weights_only=False)
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    t2s_precision = get_model_precision(t2s_model)
    t2s_model = t2s_model.to(device).to(t2s_precision)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))

def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

# if language is "zh" use the get_bert_feature, else use placeholder
def get_bert_inf(phones, word2ph, norm_text, language):
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device).to(bert_precision)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=bert_precision,
        ).to(device)

    return bert

from text import chinese
def get_phones_and_bert(text,language):
    if language == "en":
        LangSegment.setfilters(["en"])
        text = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
    elif language == "zh":
        if re.search(r'[A-Za-z]', text):
            text = re.sub(r'[a-z]', lambda x: x.group(0).upper(), text)
            text = chinese.mix_text_normalize(text)
    while "  " in text:
        text = text.replace("  ", " ")
        
    textlist=[]
    langlist=[]
    for tmp in LangSegment.getTexts(text):
        if tmp["lang"] == "en":
            langlist.append(tmp["lang"])
        else:
            langlist.append(language)
        textlist.append(tmp["text"])
    print(textlist)
    print(langlist)
    phones_list = []
    bert_list = []
    norm_text_list = []
    for i in range(len(textlist)):
        lang = langlist[i]
        phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
        print(phones,word2ph,norm_text)
        bert = get_bert_inf(phones, word2ph, norm_text, lang)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    bert = torch.cat(bert_list, dim=1)
    phones = sum(phones_list, [])
    norm_text = ''.join(norm_text_list)

    return phones,bert.to(bert_precision),norm_text

def get_tts_wav(text, top_k=20, top_p=0.6, temperature=0.6,speed=1,refer=None):
    text = text.strip("/n")
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float32
    )

    for e in splits:
        texts = text.split(e)
    audio_opt = []

    for i,text in enumerate(texts):
        text_language = "zh" #detect(text).split('-')[0] 
        phones,bert,norm_text=get_phones_and_bert(text, text_language)
        all_phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)

        with torch.no_grad():
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
            pred_semantic = pred_semantic[:, -idx:].unsqueeze(0)

        audio = (vq_model.decode(pred_semantic, torch.LongTensor(phones).to(device).unsqueeze(0),refer=refer,speed=speed).detach().cpu().numpy()[0, 0])
        max_audio=np.abs(audio).max()
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        print(audio_opt)
    return audio_opt
   
def synthesize(GPT_model_path, SoVITS_model_path, target_text_path,output_path):
    # Read target text
    with open(target_text_path, 'rb') as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        encoding = result['encoding']

    # Now open the file using the detected encoding
    with open(target_text_path, 'r', encoding=encoding) as file:
        target_text = file.read()

    # Change model weights
    change_gpt_weights(gpt_path=GPT_model_path)
    change_sovits_weights(sovits_path=SoVITS_model_path)
    refers = [get_spepc(hps, "./tools/denoised/output/DUMMY2_AZ_038.wav").to(vq_precision).to(device)]

    # Synthesize audio
    synthesis_result = get_tts_wav(text=target_text, top_p=1, temperature=1,refer=refers)

    if synthesis_result:
        concatenated_audio_data = np.concatenate(synthesis_result, axis=0)
        output_wav_path = os.path.join(output_path, os.path.basename(target_text_path)+ ".wav")
        sf.write(output_wav_path, concatenated_audio_data, hps.data.sampling_rate)
        print(f"Audio saved to {output_wav_path}")

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Command Line Tool")
    parser.add_argument('--gpt_model', required=True, help="Path to the GPT model file")
    parser.add_argument('--sovits_model', required=True, help="Path to the SoVITS model file")
    parser.add_argument('--target_text', required=True, help="Path to the target text file")
    parser.add_argument('--output_path', required=True, help="Path to the output directory")

    args = parser.parse_args()

    synthesize(args.gpt_model, args.sovits_model, args.target_text, args.output_path)

if __name__ == '__main__':
    main()

