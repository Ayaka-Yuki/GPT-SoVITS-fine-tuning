# Vtuber Voice Generator Fine-Tuning with GPT-SoVITS Pipeline

This project fine-tunes a voice generator model based on the **GPT-SoVITS pipeline** using Vtuber-related data. Below are the detailed steps of the process:

---

Here’s the complete, neatly formatted Markdown including subtitles:

## 1. Training Data Generation

- **Audio Downloading:**  
  Vtuber audio files are downloaded using [yt-dlp](https://github.com/yt-dlp/yt-dlp), along with subtitles and timestamps:

  ```bash
  yt-dlp --write-subs --all-subs -f bestaudio --extract-audio --audio-format wav --sub-format srt -o "%(title)s.%(ext)s" --cookies-from-browser chrome url
  ```

  - **For Videos Without Subtitles:**  
    [Whisper](https://github.com/openai/whisper) is used to generate subtitles in SRT format:

    ```bash
    whisper sample.wav --model small --output_format srt --language Chinese
    ```

  - **Primitive Slicing:**  
    To minimize timestamp errors, primitive slicing is performed to reduce large audio file sizes:

    ```bash
    python split_audio.py sample.wav output_folder`
```

---

## 2. Dataset Preparation

- **Slicing Audio:**  
  A custom slicer is used to cut audio files based on subtitle timestamps. However, some slices may contain long silences at the beginning or end.
  
- **Denoising:**  
  Sliced audio files undergo a denoising process to enhance quality.

- **Transcription:**  
  Corresponding text files are generated for each audio slice containing the respective transcriptions.

---

## 3. Feature Generation

### 3.1 Text Features
- Transcribed text is input into a pretrained BERT model ([Chinese-Roberta-WWM-Ext-Large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)) to generate BERT tokens.
- The text is also converted to phonemes using [pypinyin-g2pW](https://github.com/mozillazg/pypinyin-g2pW).

### 3.2 Audio Features
- Input audio is converted to CN-HuBERT features.
- CN-HuBERT features are passed to a pretrained **SynthesizerTrn** model to generate semantic features.

---

## 4. Fine-Tuning

### 4.1 SoVITS Training
- The model is trained to predict WAV audio outputs from semantic tokens.

### 4.2 GPT Training
- The GPT component is trained to predict the next semantic token using:
  - Current semantic token
  - Phonemes
  - BERT features

---

## 5. Inference

- The **inference_cli**:
  - Loads fine-tuned weights.
  - Uses a reference audio and target text to generate the target audio.

- **Observation:**  
  Removing the reference audio significantly reduces the quality of the generated target audio.

---

## 6. Evaluadtion
- We use [speechmetrics](https://github.com/aliutkus/speechmetrics/tree/master) and [mel_cepstral_distance](https://github.com/jasminsternkopf/mel_cepstral_distance)
  - Loads fine-tuned weights.
  - Uses a reference audio and target text to generate the target audio.

- **Observation:**  
  Removing the reference audio significantly reduces the quality of the generated target audio.
  
---

## Related Projects

For further details, visit the official [GPT-SoVITS GitHub Repository](https://github.com/RVC-Boss/GPT-SoVITS/tree/main?tab=readme-ov-file).

