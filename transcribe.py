"""Transcribe the audio track of a YouTube video into timestamped sentences.

Usage: python transcribe.py <url> [--language sk] [--model large-v3] [--out file.txt]
"""
import argparse
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
    sentence, start = [], None
    for segment in segments:
        for word in segment.words:
            if start is None:
                start = word.start
            sentence.append(word.word)
            text = word.word.strip()
            if SENTENCE_END.search(text) or word.end - start > MAX_SENTENCE_SECONDS:
                yield start, "".join(sentence).strip()
                sentence, start = [], None
    if sentence:
        yield start, "".join(sentence).strip()


def safe_filename(name):
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip(" .")[:120] or "transcript"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url")
    parser.add_argument("--language", help="language code, e.g. sk; detected automatically if omitted")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--out", help="output file; defaults to <video title>.txt")
    args = parser.parse_args()

    if sys.platform == "win32":
        add_cuda_dlls()

    import ctranslate2
    from faster_whisper import WhisperModel

    if ctranslate2.get_cuda_device_count() > 0:
        device, compute_type = "cuda", "float16"
    else:
        device, compute_type = "cpu", "int8"
        print("CUDA not available, running on CPU (slow).", file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        print("Downloading audio…", file=sys.stderr)
        audio, info = download_audio(args.url, tmp)

        print(f"Loading model {args.model} on {device}…", file=sys.stderr)
        model = WhisperModel(args.model, device=device, compute_type=compute_type)

        segments, details = model.transcribe(
            str(audio),
            language=args.language,
            word_timestamps=True,
            vad_filter=True,
            beam_size=5,
        )
        print(f"Language: {details.language} ({details.language_probability:.0%}), "
              f"duration {format_time(details.duration)}", file=sys.stderr)

        out = Path(args.out or safe_filename(info.get("title") or info["id"]) + ".txt")
        with out.open("w", encoding="utf-8") as f:
            for start, text in to_sentences(segments):
                line = f"{format_time(start)} {text}"
                f.write(line + "\n")
                print(f"\r{format_time(start)} / {format_time(details.duration)}", end="", file=sys.stderr)

    print(f"\nSaved to {out.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
