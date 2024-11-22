import subprocess

# Define the URL and language
url = "https://space.bilibili.com/30331740/video"
language_code = "zh"  # Replace with desired subtitle language code

# Construct the yt-dlp command
command = [
    "yt-dlp", 
    "--write-sub", 
    f"--sub-lang={language_code}", 
    "--skip-download", 
    "--sub-format", 
    "srt", 
    url
]

# Execute the command
subprocess.run(command)
