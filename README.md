# analyze-video

Transcribes the audio track of a YouTube video into timestamped sentences, locally on the GPU.

```
0:04 First sentence.
10:08 Another sentence.
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install "yt-dlp[default]" faster-whisper nvidia-cublas-cu12 nvidia-cudnn-cu12
```

YouTube downloads also need a JavaScript runtime on `PATH` ([Deno](https://deno.com) or Node.js 22+).

For speaker diarization (`--diarize`) also:

```bash
.venv\Scripts\python.exe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install pyannote.audio --extra-index-url https://download.pytorch.org/whl/cu128
```

Then accept the terms of [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) and put a Hugging Face read token in the `HF_TOKEN` environment variable.

## Usage

```bash
.venv\Scripts\python.exe transcribe.py "<youtube url>" [--language sk] [--out file.txt]
```

With speaker labels:

```bash
.venv\Scripts\python.exe transcribe.py "<youtube url>" --diarize [--speakers 2 | --min-speakers 2 --max-speakers 4]
```

```
0:04 [Speaker 1] First sentence.
0:09 [Speaker 2] Reply.
```
