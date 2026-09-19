import os
import sys
from faster_whisper import WhisperModel

import datetime 
from os import listdir
from os.path import isfile, join

from tqdm import tqdm


def get_whisper_model(
    model_size: str = "base", dry_run: bool = True
) -> WhisperModel:

    if dry_run:
        print("Running in DRY-RUN mode (CPU, int8 quantization)...")
        return WhisperModel(
            model_size_or_path=model_size, device="cpu", compute_type="int8"
        )

    print("Running in PRODUCTION mode (GPU/CUDA, float16)...")
    try:
        return WhisperModel(
            model_size_or_path=model_size, device="cuda", compute_type="float16"
        )
    except Exception as e:
        print(f"Failed to initialize GPU model: {e}", file=sys.stderr)
        print("Falling back to CPU.", file=sys.stderr)
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

def get_timestamp():
    now = datetime.datetime.now()
    return now.strftime('%Y%m%d%H%M%S') 

if __name__ == "__main__":

    is_dry_run = True

    model_name = "tiny" if is_dry_run else "large-v3"
    output_dir = f"data/{model_name}-{get_timestamp()}"
    input_dir = "data/chat_source"

    files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
    opus_files = []
    for f in files:
        parts = f.split(".")
        if len(parts) == 2 and parts[1] == 'opus':
            opus_files.append(f)

    os.makedirs(output_dir, exist_ok=True)
    model = get_whisper_model(model_size=model_name, dry_run=is_dry_run)

    for file in tqdm(opus_files):
        in_path = join(input_dir, file)
        text = transcribe_audio(model, in_path)
        txt_filename = file.split(".")[0] + ".txt"
        out_path = join(output_dir, txt_filename) 
        with open(out_path, "w") as f:
            f.write(text)
