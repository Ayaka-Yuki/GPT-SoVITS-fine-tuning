from text import cleaned_text_to_sequence
from text import symbols as symbols_v1
import os
import sys
sys.path.append(os.getcwd())


special = [
    ("￥", "zh", "SP2"),
    ("^", "zh", "SP3"),
]


def clean_text(text, language):
    symbols = symbols_v1.symbols
    language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}

    if(language not in language_module_map):
        language="en"
        text=" "
    for special_s, special_l, target_symbol in special:
        if special_s in text and language == special_l:
            return clean_special(text, language, special_s, target_symbol)
    try:
        language_module = __import__("text."+language_module_map[language], fromlist=[language_module_map[language]])
    except ImportError as e:
        print(f"Error: Module for language {language} not found. Exception: {e}")

    if hasattr(language_module,"text_normalize"):
        norm_text = language_module.text_normalize(text)
    else:
        norm_text=text
    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    elif language == "en":
        phones = language_module.g2p(norm_text)
        if len(phones) < 4:
            phones = [','] + phones
        word2ph = None
    else:
        phones = language_module.g2p(norm_text)
        word2ph = None
    phones = ['UNK' if ph not in symbols else ph for ph in phones]
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    symbols = symbols_v1.symbols
    language_module_map = {"zh": "chinese", "ja": "japanese", "en": "english"}

    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = __import__("text."+language_module_map[language],fromlist=[language_module_map[language]])
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊光缺、还是到付红四方。", "zh"))
