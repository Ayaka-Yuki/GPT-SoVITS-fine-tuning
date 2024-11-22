url = "custom_url"

# find videos from (https://space.bilibili.com/30331740/) from 阿梓

# check available subtitles
yt-dlp --list-subs --cookies-from-browser chrome url

# download subtitles for downstream audio slicing
yt-dlp --write-subs --all-subs --skip-download --sub-format srt -o "%(title)s.%(ext)s" --cookies-from-browser chrome url

# download its audio file along with subtitle
yt-dlp -f bestaudio --extract-audio --audio-format wav --cookies-from-browser chrome url

# We need subtitle for slicing and audio as its input


