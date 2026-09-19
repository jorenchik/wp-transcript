
import re
from os.path import isfile, join

pattern = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}),\s"  # 1. Date (e.g., 4/7/26)
    r"(\d{2}:\d{2})\s-\s"             # 2. Time (e.g., 20:03)
    r"([^:]+):\s"                     # 3. Sender (anything up to the colon)
    r"(.+?)\s"                        # 4. Filename (lazy match)
    r"\(file attached\)$"             # 5. Literal string anchor
)

def get_transcription(transcription_path, opus_filename):
    txt_filename = opus_filename.split(".")[0] + ".txt"
    txt_path = join(transcription_path, txt_filename)
    text = None
    with open(txt_path, "r") as f:
        text = f.read()
    return text

if __name__ == "__main__":

    input_file = "data/chat_source/chat_log.txt"
    transcription_path = "data/tiny-20260801092854";
    output_file = join(transcription_path, "chat_out.txt")

    text = None;
    with open(input_file, "r") as f:
        text = f.read()

    lines = text.split("\n")
    with open(output_file, "w") as f:
        for line in lines:

            if line == '':
                continue

            match = pattern.match(line)
            output_line = None
            if match:
                date, time, sender, filename = match.groups()
                extension = filename.split(".")[1]
                content = None
                if extension == 'opus':
                    content = "[Transcribed] " + get_transcription(transcription_path, filename)
                else:
                    content = f"{filename} (file attached)"
                output_line = f"{date}, {time} - {sender}: {content}"
            else:
                output_line = line

            f.write(output_line + "\n")





