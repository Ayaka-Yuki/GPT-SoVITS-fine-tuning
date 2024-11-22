import os
import argparse
from pydub import AudioSegment

def split_audio_max_length(input_file, output_folder, max_length_ms=180000):
    """
    Splits a large .wav file into smaller chunks based on a maximum length of 10 minutes.
    
    :param input_file: Path to the input .wav file
    :param output_folder: Folder to save the split .wav files
    :param max_length_ms: Maximum length of each chunk in milliseconds (default 600000ms = 10 minutes)
    """
    
    # Load the audio file
    audio = AudioSegment.from_wav(input_file)

    # Calculate the number of chunks
    num_chunks = len(audio) // max_length_ms + (1 if len(audio) % max_length_ms != 0 else 0)

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Export each chunk as a separate .wav file
    for i in range(num_chunks):
        start_ms = i * max_length_ms
        end_ms = min((i + 1) * max_length_ms, len(audio))  # Ensure we don't go beyond the audio's length
        
        chunk = audio[start_ms:end_ms]
        
        output_file = os.path.join(output_folder, f"chunk_{i+1}.wav")
        chunk.export(output_file, format="wav")
        print(f"Saved chunk {i+1} as {output_file}")

if __name__ == "__main__":
    # Setup the argument parser
    parser = argparse.ArgumentParser(description="Split a large .wav file into smaller chunks")
    
    # Add arguments
    parser.add_argument("input_file", type=str, help="Path to the input .wav file")
    parser.add_argument("output_folder", type=str, help="Folder to save the resulting smaller .wav files")
    parser.add_argument("--max_length", type=int, default=180000, help="Maximum length for each chunk in milliseconds (default 600000ms = 10 minutes)")
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the parsed arguments
    split_audio_max_length(args.input_file, args.output_folder, args.max_length)
