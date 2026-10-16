"""Transcribe the audio track of a YouTube video into timestamped sentences.

Usage: python transcribe.py <url> [--language sk] [--model large-v3] [--out file.txt]
                            [--diarize [--speakers N | --min-speakers N --max-speakers N]]
"""
import argparse
import bisect
import os
import re
import sys
import sysconfig
import tempfile
from pathlib import Path

SENTENCE_END = re.compile(r'[.!?…]["\')\]]*$')
MAX_SENTENCE_SECONDS = 30


def add_cuda_dlls():
    # pip-installed NVIDIA libraries are not on the DLL search path on Windows.
    nvidia = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    for bin_dir in nvidia.glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ["PATH"]


def download_audio(url, target_dir):
    import yt_dlp

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(Path(target_dir) / "%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        # YouTube requires solving JS challenges; without a runtime downloads can fail with HTTP 403.
        "js_runtimes": {"deno": {}, "node": {}},
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)), info


def format_time(seconds):
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def to_sentences(segments):
    words = []
    for segment in segments:
        for word in segment.words:
            words.append(word)
            if SENTENCE_END.search(word.word.strip()) or word.end - words[0].start > MAX_SENTENCE_SECONDS:
                yield words
                words = []
    if words:
        yield words


def diarize(audio, sample_rate, device, num_speakers, min_speakers, max_speakers):
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    import torch
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set; diarization needs a Hugging Face token.")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    pipeline.to(torch.device(device))
    output = pipeline(
        {"waveform": torch.from_numpy(audio)[None], "sample_rate": sample_rate},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    # The exclusive variant has no overlapping turns, which suits assigning words to speakers.
    annotation = output.exclusive_speaker_diarization
    return sorted((turn.start, turn.end, label) for turn, _, label in annotation.itertracks(yield_label=True))


class SpeakerLabels:
    """Assigns sentences to diarized speakers, numbered by first appearance."""

    def __init__(self, turns):
        self.turns = turns
        self.starts = [start for start, _, _ in turns]
        self.numbers = {}

    def speaker_at(self, time):
        i = bisect.bisect_right(self.starts, time) - 1
        candidates = [self.turns[j] for j in (i, i + 1) if 0 <= j < len(self.turns)]
        if not candidates:
            return None
        # The turn containing the time, otherwise the nearest one.
        return min(candidates, key=lambda t: 0 if t[0] <= time <= t[1] else min(abs(time - t[0]), abs(time - t[1])))[2]

    def label(self, words):
        durations = {}
        for word in words:
            speaker = self.speaker_at((word.start + word.end) / 2)
            durations[speaker] = durations.get(speaker, 0) + (word.end - word.start)
        speaker = max(durations, key=durations.get)
        if speaker is None:
            return "[?]"
        number = self.numbers.setdefault(speaker, len(self.numbers) + 1)
        return f"[Speaker {number}]"


def safe_filename(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:120] or "transcript"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--language", help="language code, e.g. sk; detected automatically if omitted")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--out", help="output file; defaults to <video title>.txt")
    parser.add_argument("--diarize", action="store_true", help="label each sentence with its speaker")
    parser.add_argument("--speakers", type=int, help="exact number of speakers, if known")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    args = parser.parse_args()

    if sys.platform == "win32":
        add_cuda_dlls()

    import ctranslate2
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    if ctranslate2.get_cuda_device_count() > 0:
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"
        print("CUDA not available, running on CPU (slow).", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        print("Downloading audio…", file=sys.stderr)
        audio_file, info = download_audio(args.url, tmp)
        sample_rate = 16000
        audio = decode_audio(str(audio_file), sampling_rate=sample_rate)

        speakers = None
        if args.diarize:
            print("Diarizing speakers…", file=sys.stderr)
            turns = diarize(audio, sample_rate, device, args.speakers, args.min_speakers, args.max_speakers)
            speakers = SpeakerLabels(turns)
            print(f"Found {len({label for _, _, label in turns})} speakers.", file=sys.stderr)

        print(f"Loading model {args.model} on {device}…", file=sys.stderr)
        model = WhisperModel(args.model, device=device, compute_type=compute_type)

        segments, details = model.transcribe(
            audio,
            language=args.language,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
        print(f"Language: {details.language} ({details.language_probability:.0%}), "
              f"duration {format_time(details.duration)}", file=sys.stderr)

        out = Path(args.out or safe_filename(info.get("title") or info["id"]) + ".txt")
        with out.open("w", encoding="utf-8") as f:
            for words in to_sentences(segments):
                start = format_time(words[0].start)
                text = "".join(word.word for word in words).strip()
                line = f"{start} {speakers.label(words)} {text}" if speakers else f"{start} {text}"
                f.write(line + "\n")
                print(f"\r{start} / {format_time(details.duration)}", end="", file=sys.stderr)

    print(f"\nSaved to {out.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
