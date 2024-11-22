import os
from pydub import AudioSegment
import pysrt
from tqdm import tqdm
import librosa
import re

def get_sample_rate_librosa(audio_file):
    # Load the audio file with librosa (it returns audio and the sample rate)
    y, sr = librosa.load(audio_file, sr=None)  # sr=None ensures we don't resample the audio
    return sr

def slice_audio_dynamic(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Initialize counter for DUMMY folders
    dummy_counter = 0

    # Iterate through all MP3 files in the input folder
    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            wav_file = os.path.join(input_folder, file)
            base_name = os.path.splitext(file)[0]
            srt_file = os.path.join(input_folder, f"{base_name}.srt")

            # Skip if matching SRT file is not found
            if not os.path.exists(srt_file):
                print(f"Matching SRT file not found for: {wav_file}")
                continue

            print(f"Processing: {wav_file} with {srt_file}")

            audio = AudioSegment.from_wav(wav_file)

            # Parse the subtitle file
            subs = pysrt.open(srt_file)

            dummy_counter += 1

            # List to store phonetic transcriptions
            final_output = []

            # Slice the audio based on subtitles
            for i, sub in tqdm(enumerate(subs), total=len(subs), desc=f"Processing slices for {base_name}"):
                # Skip subtitles with music symbols ♪
                if '♪' in sub.text:
                    continue

                if re.search(r'[a-zA-Z]', sub.text):
                    continue

                if len(sub.text.strip()) < 2:
                    continue

                # Convert start and end time to milliseconds
                start_ms = (sub.start.hours * 3600 + sub.start.minutes * 60 + sub.start.seconds) * 1000 + sub.start.milliseconds
                end_ms = (sub.end.hours * 3600 + sub.end.minutes * 60 + sub.end.seconds) * 1000 + sub.end.milliseconds
                
                # Slice the audio
                sliced_audio = audio[start_ms:end_ms]

                output_audio_name = f"DUMMY{dummy_counter}_AZ_{str(i+1).zfill(3)}.wav"
                output_audio_path = os.path.join(output_folder, output_audio_name)
                sliced_audio.export(output_audio_path, format="wav")

                # Append transcription data
                final_output.append(f"{output_audio_path}|AZI|ZH|{sub.text}。")

            # Save transcription to a single file
            phonetic_file_path = os.path.join(output_folder, "final_phonetic_transcriptions.txt")
            with open(phonetic_file_path, 'a', encoding='utf-8') as phonetic_file:
                for line in final_output:
                    phonetic_file.write(line + "\n")

            print(f"Finished processing {wav_file}. Transcriptions saved to {phonetic_file_path}")

# Example usage
cur = os.getcwd()
input_folder = os.path.join(cur, "audio_text_folder") 
print(input_folder)
output_folder = './output'    # Output folder for sliced audio and transcriptions


slice_audio_dynamic(input_folder, output_folder)