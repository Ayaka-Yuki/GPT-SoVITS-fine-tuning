from pypinyin import pinyin, lazy_pinyin, Style
from pypinyin.contrib.tone_convert import to_normal, to_tone, to_initials, to_finals

def convert_to_pinyin(text):
    # Convert the text to Pinyin with tone marks
    pinyin_list = pinyin(text, style=Style.TONE3,neutral_tone_with_five=True)  # Style.TONE3 includes the tone marks (e.g., "ma1", "ma2")
    
    pinyin_text = []
    
    for word in pinyin_list:
        for p in word:
            # Extract the full pinyin with tone marks
            initial = to_initials(p)  # Initial consonant
            final = to_finals(p)  # Final vowel sound
            tone = p[-1]  # Last character will be the tone number (e.g., "ma1" -> "1")
            
            if not initial:
                initial = "~"  # Representing no initial consonant with "~"
            
            # Append the result in the format: Initial Final Tone
            pinyin_text.append(f"{initial} {final}{tone}")
    
    return " ".join(pinyin_text)

# Example usage
text = "你好，世界"
pinyin_result = convert_to_pinyin(text)
print(pinyin_result)
