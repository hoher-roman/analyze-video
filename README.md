# analyze-video

Transcribes the audio track of a YouTube video into timestamped sentences, locally on the GPU.

```
0:04 First sentence.
10:08 Another sentence.
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install yt-dlp faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
```

## Usage

```bash
.venv\Scripts\python.exe transcribe.py "<youtube url>" [--language sk] [--out file.txt]
```
