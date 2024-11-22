import opencc
import sys

# Ensure correct usage
if len(sys.argv) != 3:
    print("Usage: python convert_srt.py <input_file.srt> <output_file.srt>")
    sys.exit(1)

# Get input and output file paths from arguments
input_srt = sys.argv[1]
output_srt = sys.argv[2]

# Initialize OpenCC converter for Traditional to Simplified Chinese
converter = opencc.OpenCC('t2s')

# Read the input SRT file
with open(input_srt, "r", encoding="utf-8") as file:
    lines = file.readlines()

# Convert Traditional Chinese to Simplified Chinese line by line
converted_lines = [converter.convert(line) for line in lines]

# Write the converted content to the output SRT file
with open(output_srt, "w", encoding="utf-8") as file:
    file.writelines(converted_lines)

print(f"Converted SRT saved to {output_srt}")

