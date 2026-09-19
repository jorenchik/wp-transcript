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


def parse_chat_datetime(date_str: str, time_str: str) -> datetime.datetime:
    """
    Parse WhatsApp-style date (e.g. '4/7/26' or '12/9/2026') + time ('20:03')
    into a datetime object. Assumes day/month/year ordering as in the sample
    export; adjust if your export uses month/day/year.
    """
    day, month, year = date_str.split("/")
    day, month = int(day), int(month)
    year = int(year)
    if year < 100:
        year += 2000
    hour, minute = map(int, time_str.split(":"))
    return datetime.datetime(year, month, day, hour, minute)


def parse_cli_datetime(value: str) -> datetime.datetime:
    """Parse CLI-provided date filters in format YYYY-MM-DDTHH:MM."""
    return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M")


# ---------------------------------------------------------------------------
# Transcription logic
# ---------------------------------------------------------------------------

def get_whisper_model(
    model_size: str = "base",
    dry_run: bool = True,
    compute_type_override: str = None,
    cpu_threads: int = 0,
    num_workers: int = 1,
) -> WhisperModel:

    if dry_run:
        compute_type = compute_type_override or "int8"
        click.echo(f"Running in DRY-RUN mode (CPU, {compute_type} quantization)...")
        return WhisperModel(
            model_size_or_path=model_size,
            device="cpu",
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )

    compute_type = compute_type_override or "float16"
    click.echo(f"Running in PRODUCTION mode (GPU/CUDA, {compute_type})...")
    try:
        return WhisperModel(
            model_size_or_path=model_size,
            device="cuda",
            compute_type=compute_type,
            num_workers=num_workers,
        )
    except Exception as e:
        click.echo(f"Failed to initialize GPU model: {e}", err=True)
        click.echo("Falling back to CPU.", err=True)
        return WhisperModel(
            model_size_or_path=model_size,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            num_workers=num_workers,
        )


def transcribe_audio(
    model: WhisperModel,
    audio_path: str,
    beam_size: int = 5,
    vad_filter: bool = False,
) -> str:

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    segments, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        vad_filter=vad_filter,
    )
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
    "--source-dir",
    "source_dir",
    default="data/chat_source",
    show_default=True,
    help="Directory containing .opus audio files (the chat source directory).",
)
@click.option(
    "--output-base-dir",
    "output_base_dir",
    default="data",
    show_default=True,
    help="Base directory under which the generated output directory "
         "(named <model_name>-<timestamp>) will be created.",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    help="Explicit directory to write .txt transcriptions to. "
         "If not set, defaults to <output-base-dir>/<model_name>-<timestamp>.",
)
@click.option(
    "--dry-run/--no-dry-run",
    default=True,
    show_default=True,
    help="Use tiny/CPU model for a quick dry run instead of large-v3/GPU.",
)
@click.option(
    "--beam-size",
    default=5,
    show_default=True,
    type=int,
    help="Beam size for decoding. Use 1 for greedy (fastest, lower quality).",
)
@click.option(
    "--vad-filter/--no-vad-filter",
    default=True,
    show_default=True,
    help="Skip silence using voice activity detection (usually faster, minimal quality impact).",
)
@click.option(
    "--compute-type",
    default=None,
    help="Override compute type, e.g. int8, int8_float16, float16, float32.",
)
@click.option(
    "--cpu-threads",
    default=0,
    show_default=True,
    type=int,
    help="Number of CPU threads to use (0 = let faster-whisper decide).",
)
@click.option(
    "--num-workers",
    default=1,
    show_default=True,
    type=int,
    help="Number of parallel workers for the model (useful for batch decoding).",
)
@click.option(
    "--jobs",
    default=1,
    show_default=True,
    type=int,
    help="Number of files to transcribe concurrently using a thread pool.",
)
def transcribe(
    source_dir,
    output_base_dir,
    output_dir,
    dry_run,
    beam_size,
    vad_filter,
    compute_type,
    cpu_threads,
    num_workers,
    jobs,
):
    """Transcribe .opus files in SOURCE_DIR to .txt files."""
    model_name = "tiny" if dry_run else "large-v3"

    if output_dir is None:
        output_dir = join(output_base_dir, f"{model_name}-{get_timestamp()}")

    files = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
    opus_files = [f for f in files if f.split(".")[-1] == "opus" and len(f.split(".")) == 2]

    os.makedirs(output_dir, exist_ok=True)
    model = get_whisper_model(
        model_size=model_name,
        dry_run=dry_run,
        compute_type_override=compute_type,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
    )

    def process(file):
        in_path = join(source_dir, file)
        text = transcribe_audio(model, in_path, beam_size=beam_size, vad_filter=vad_filter)
        txt_filename = file.split(".")[0] + ".txt"
        out_path = join(output_dir, txt_filename)
        with open(out_path, "w") as f:
            f.write(text)

    if jobs > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {executor.submit(process, f): f for f in opus_files}
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()
    else:
        for file in tqdm(opus_files):
            process(file)

    click.echo(f"Done. Transcriptions written to: {output_dir}")


@cli.command(name="replace-audios")
@click.option(
    "--source-dir",
    "source_dir",
    default="data/chat_source",
    show_default=True,
    help="Directory containing the raw WhatsApp chat log file (and originally-referenced media).",
)
@click.option(
    "--input-file",
    "input_file",
    default=None,
    help="Path to the raw WhatsApp chat log file. "
         "Defaults to <source-dir>/chat_log.txt.",
)
@click.option(
    "--input-filename",
    "input_filename",
    default="chat_log.txt",
    show_default=True,
    help="Filename of the chat log inside --source-dir "
         "(used only if --input-file is not explicitly set).",
)
@click.option(
    "--transcription-path",
    "transcription_path",
    required=True,
    help="Directory containing the transcribed .txt files (output of `transcribe`).",
)
@click.option(
    "--output-dir",
    "output_dir",
    default=None,
    help="Directory to write the output chat log into. "
         "Defaults to --transcription-path.",
)
@click.option(
    "--output-file",
    "output_file",
    default=None,
    help="Path to write the output chat log with transcriptions. "
         "Defaults to <output-dir>/chat_out.txt.",
)
@click.option(
    "--output-filename",
    "output_filename",
    default="chat_out.txt",
    show_default=True,
    help="Filename for the output chat log inside --output-dir "
         "(used only if --output-file is not explicitly set).",
)
@click.option(
    "--from-date",
    "from_date",
    default=None,
    help="Only include lines at/after this datetime, format YYYY-MM-DDTHH:MM.",
)
@click.option(
    "--to-date",
    "to_date",
    default=None,
    help="Only include lines at/before this datetime, format YYYY-MM-DDTHH:MM.",
)
def replace_audios(
    source_dir,
    input_file,
    input_filename,
    transcription_path,
    output_dir,
    output_file,
    output_filename,
    from_date,
    to_date,
):
    """Replace '(file attached)' opus references in a chat log with transcriptions."""
    if input_file is None:
        input_file = join(source_dir, input_filename)

    if output_dir is None:
        output_dir = transcription_path

    if output_file is None:
        output_file = join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    from_dt = parse_cli_datetime(from_date) if from_date else None
    to_dt = parse_cli_datetime(to_date) if to_date else None

    with open(input_file, "r") as f:
        text = f.read()

    lines = text.split("\n")
    with open(output_file, "w") as f:
        for line in lines:
            if line == '':
                continue

            match = PATTERN.match(line)
            output_line = None

            # Only lines matching the "(file attached)" pattern carry a
            # reliably parseable date/time header in this implementation.
            # Non-matching lines (e.g. continuation lines / plain text
            # messages) are passed through unless we can also extract a
            # leading date/time from them.
            date, time = None, None
            if match:
                date, time, sender, filename = match.groups()
            else:
                # Try to extract a leading "date, time - sender:" prefix
                # from plain text lines too, so filtering applies uniformly.
                plain_match = re.match(
                    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{2}:\d{2})\s-\s", line
                )
                if plain_match:
                    date, time = plain_match.groups()

            if date and time and (from_dt or to_dt):
                try:
                    line_dt = parse_chat_datetime(date, time)
                except ValueError:
                    line_dt = None

                if line_dt is not None:
                    if from_dt and line_dt < from_dt:
                        continue
                    if to_dt and line_dt > to_dt:
                        continue

            if match:
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
