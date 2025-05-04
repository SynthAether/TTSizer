import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json

load_dotenv()

    
FILE_PATH = "temp/vocals/aot_e10_vocals.wav"
OUTPUT_JSON = "temp/output5.json"
# MODEL = "gemini-2.5-pro-preview-03-25"
MODEL = "gemini-2.5-pro-exp-03-25"
TEMPERATURE=0.1

with open('prompt.txt', 'r') as f:
    PROMPT = f.read()

def generate():
    client = genai.Client(
        api_key=os.environ.get("GEMINI_API_KEY"),
    )
    
    print(f"uploading file")
    files = [client.files.upload(file=FILE_PATH)]
    print(f"file upload complete")
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_uri(
                    file_uri=files[0].uri,
                    mime_type=files[0].mime_type,
                ),
                types.Part.from_text(text=PROMPT),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        temperature=0.1,
        safety_settings=[
            types.SafetySetting(
                category="HARM_CATEGORY_HARASSMENT",
                threshold="BLOCK_NONE",  
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_HATE_SPEECH",
                threshold="BLOCK_NONE",  
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                threshold="BLOCK_NONE", 
            ),
            types.SafetySetting(
                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                threshold="BLOCK_NONE",  
            ),
        ],
        response_mime_type="application/json",
    )
        
    response = client.models.generate_content(
        model=MODEL,
        contents=contents,
        config=generate_content_config,
    )
    extracted_json_data = response.candidates[0].content.parts[0].model_dump()['text']
    extracted_json_data = json.loads(extracted_json_data)
                
    with open(OUTPUT_JSON, 'w', encoding="utf-8") as f:
        json.dump(extracted_json_data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    generate()
