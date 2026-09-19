#!/usr/bin/env python3
import os
import sys
import re
import datetime
from os import listdir
from os.path import isfile, join

import click
from tqdm import tqdm
from faster_whisper import WhisperModel


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def get_timestamp():
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d%H%M%S')


PATTERN = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s"  # 1. Date (e.g., 4/7/26)
    r"(\d{2}:\d{2})\s-\s"             # 2. Time (e.g., 20:03)
    r"([^:]+):\s"                     # 3. Sender (anything up to the colon)
    r"(.+?)\s"                        # 4. Filename (lazy match)
    r"\(file attached\)$"             # 5. Literal string anchor
)


# ---------------------------------------------------------------------------
# Transcription logic
# ---------------------------------------------------------------------------

def get_whisper_model(model_size: str = "base", dry_run: bool = True) -> WhisperModel:
    if dry_run:
        click.echo("Running in DRY-RUN mode (CPU, int8 quantization)...")
        return WhisperModel(
            model_size_or_path=model_size, device="cpu", compute_type="int8"
        )

    click.echo("Running in PRODUCTION mode (GPU/CUDA, float16)...")
    try:
        return WhisperModel(
            model_size_or_path=model_size, device="cuda", compute_type="float16"
        )
    except Exception as e:
        click.echo(f"Failed to initialize GPU model: {e}", err=True)
        click.echo("Falling back to CPU.", err=True)
        return WhisperModel(
            model_size_or_path=model_size, device="cpu", compute_type="int8"
        )


def transcribe_audio(model: WhisperModel, audio_path: str) -> str:
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    segments, info = model.transcribe(audio_path, beam_size=5)
    text = ""
    for segment in segments:
        text += segment.text

    if len(text) > 0 and text[0] == ' ':
        text = text[1:]

    return text


# ---------------------------------------------------------------------------
# Replace-audios logic
# ---------------------------------------------------------------------------

def get_transcription(transcription_path, opus_filename):
    txt_filename = opus_filename.split(".")[0] + ".txt"
    txt_path = join(transcription_path, txt_filename)
    with open(txt_path, "r") as f:
        text = f.read()
    return text


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Chat audio transcription toolkit."""
    pass


@cli.command()
@click.option(
    "--input-dir",
    default="data/chat_source",
    show_default=True,
    help="Directory containing .opus audio files.",
)
@click.option(
    "--output-dir",
    default=None,
    help="Directory to write .txt transcriptions to. "
         "Defaults to data/<model_name>-<timestamp>.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    show_default=True,
    help="Use tiny/CPU model for a quick dry run instead of large-v3/GPU.",
)
def transcribe(input_dir, output_dir, dry_run):
    """Transcribe .opus files in INPUT_DIR to .txt files."""
    model_name = "tiny" if dry_run else "large-v3"

    if output_dir is None:
        output_dir = f"data/{model_name}-{get_timestamp()}"

    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    opus_files = [f for f in files if f.split(".")[-1] == "opus" and len(f.split(".")) == 2]

    os.makedirs(output_dir, exist_ok=True)
    model = get_whisper_model(model_size=model_name, dry_run=dry_run)

    for file in tqdm(opus_files):
        in_path = join(input_dir, file)
        text = transcribe_audio(model, in_path)
        txt_filename = file.split(".")[0] + ".txt"
        out_path = join(output_dir, txt_filename)
        with open(out_path, "w") as f:
            f.write(text)

    click.echo(f"Done. Transcriptions written to: {output_dir}")


@cli.command(name="replace-audios")
@click.option(
    "--input-file",
    default="data/chat_source/chat_log.txt",
    show_default=True,
    help="Path to the raw WhatsApp chat log file.",
)
@click.option(
    "--transcription-path",
    required=True,
    help="Directory containing the transcribed .txt files (output of `transcribe`).",
)
@click.option(
    "--output-file",
    default=None,
    help="Path to write the output chat log with transcriptions. "
         "Defaults to <transcription-path>/chat_out.txt.",
)
def replace_audios(input_file, transcription_path, output_file):
    """Replace '(file attached)' opus references in a chat log with transcriptions."""
    if output_file is None:
        output_file = join(transcription_path, "chat_out.txt")

    with open(input_file, "r") as f:
        text = f.read()

    lines = text.split("\n")
    with open(output_file, "w") as f:
        for line in lines:
            if line == '':
                continue

            match = PATTERN.match(line)
            if match:
                date, time, sender, filename = match.groups()
                extension = filename.split(".")[1]
                if extension == 'opus':
                    content = "[Transcribed] " + get_transcription(transcription_path, filename)
                else:
                    content = f"{filename} (file attached)"
                output_line = f"{date}, {time} - {sender}: {content}"
            else:
                output_line = line

            f.write(output_line + "\n")

    click.echo(f"Done. Output written to: {output_file}")


if __name__ == "__main__":
    cli()
