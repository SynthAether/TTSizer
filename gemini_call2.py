import os
from google import genai
from google.genai import types
import json
from pathlib import Path
from typing import List
from tqdm import tqdm

MODEL = "gemini-2.5-pro-exp-03-25"
TEMPERATURE = 0.1
PROMPT_TEMPLATE_PATH = Path('prompt_template.txt')

def build_context_and_prompt(anime_title: str, characters_of_interest: List[str]) -> str:
    """
    Constructs the BASE_PROMPT based on input anime title and character list.
    """
    template_text = PROMPT_TEMPLATE_PATH.read_text(encoding='utf-8')
    template_chars = "[" + ", ".join(characters_of_interest) + "]"
    base_prompt = (
        template_text
        .replace("{ANIME_TITLE}", anime_title)
        .replace("{CHARACTERS_OF_INTEREST}", template_chars)
        .replace("{CHARACTER_1}", characters_of_interest[0])
        .replace("{CHARACTER_2}", characters_of_interest[1])
    )
    return base_prompt

def generate_on_file(flac_path: Path, output_json: Path, base_prompt: str):
    """
    Uploads the FLAC file and runs Gemini diarization, writes JSON result.
    """
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    print(f"Uploading {flac_path.name}...")
    uploaded = client.files.upload(file=str(flac_path))
    print(f"Uploaded: {uploaded.uri}")

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=uploaded.uri,
                    mime_type=uploaded.mime_type,
                ),
                types.Part.from_text(text=base_prompt),
            ],
        )
    ]
    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        safety_settings=[
            types.SafetySetting(category=c, threshold="BLOCK_NONE")
            for c in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ],
        response_mime_type="application/json",
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=config,
    )
    json_text = response.candidates[0].content.parts[0].model_dump()["text"]
    data = json.loads(json_text)

    output_json.write_text(json.dumps(data, indent=4, ensure_ascii=False), encoding='utf-8')
    print(f"Wrote output to {output_json.name}")

def batch_generate(anime_title: str, characters_of_interest: List[str], input_dir: Path, output_dir: Path):
    """
    Process all .flac files in input_dir and write corresponding JSON in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    base_prompt = build_context_and_prompt(anime_title, characters_of_interest)

    flac_files = sorted(input_dir.glob("*.flac"))
    if not flac_files:
        print(f"No .flac files found in {input_dir}")
        return

    for fl in tqdm(flac_files):
        out_json = output_dir / (fl.stem + ".json")
        try:
            generate_on_file(fl, out_json, base_prompt)
        except Exception as e:
            print(f"Error processing {fl.name}: {e}")

if __name__ == "__main__":
    anime_title = "Attack on Titan Season 1 Episode 10"
    characters_of_interest = [
        "Eren Yeager",
        "Mikasa Ackerman",
        "Armin Arlert",
        "Levi Ackerman"
    ]
    input_dir = Path("temp/vocals_flac/")
    output_dir = Path("temp/outputs_json/")

    batch_generate(anime_title, characters_of_interest, input_dir, output_dir)
