import os
import subprocess
import argparse

def transcribe_all_chunks(input_folder, model="small", output_format="srt", language="Chinese"):
    # List all .wav files in the input folder
    wav_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]
    
    if not wav_files:
        print("No .wav files found in the folder.")
        return
    
    # Process each .wav file
    for wav_file in sorted(wav_files):  # Sort to maintain chunk order
        input_path = os.path.join(input_folder, wav_file)
        output_path = os.path.join(input_folder, f"{os.path.splitext(wav_file)[0]}.{output_format}")
        
        # Run Whisper command
        command = [
            "whisper",
            input_path,
            "--model", model,
            "--output_format", output_format,
            "--language", language,
            "--output_dir", input_folder  # Save output in the same folder
        ]
        print(f"Processing: {wav_file}")
        try:
            subprocess.run(command, check=True)
            print(f"Output saved to: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {wav_file}: {e}")

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Transcribe all .wav files in a folder using Whisper.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing .wav files")
    parser.add_argument("--model", type=str, default="small", help="Whisper model to use (default: 'small')")
    parser.add_argument("--output_format", type=str, default="srt", help="Output format for transcription (default: 'srt')")
    parser.add_argument("--language", type=str, default="Chinese", help="Language of the audio (default: 'Chinese')")
    
    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    transcribe_all_chunks(
        input_folder=args.input_folder,
        model=args.model,
        output_format=args.output_format,
        language=args.language
    )
