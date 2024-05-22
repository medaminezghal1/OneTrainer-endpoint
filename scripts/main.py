import sys

sys.path.append('/home/OneTrainer/scripts')


from train import main as train_main

from fastapi import FastAPI, File, Query, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import subprocess
import json
import os
import csv
from urllib.request import urlopen
import base64
import tempfile
import shutil

app = FastAPI()

class ImageTextData:
    def __init__(self, image_url, image_data, text):
        self.image_url = image_url
        self.image_data = image_data
        self.text = text

def download_and_store(csv_file_path):
    image_data_list = []
    with open(csv_file_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row (assuming there's a header)
        for row in reader:
            try:
                image_url, text = row[0], row[1]
                response = urlopen(image_url)
                image_data = base64.b64encode(response.read()).decode('utf-8')
                image_data_list.append(ImageTextData(image_url, image_data, text))
            except Exception as e:
                print(f"Error downloading {image_url}: {e}")
    return image_data_list

@app.post("/train_lora")
async def train_lora(file: UploadFile, param: str = Query(...)):
    json_file_path = "/home/OneTrainer/training_presets/sdxl_1.0_LoRA_style.json"
    temp_dir = tempfile.mkdtemp()
    temp_file_path = os.path.join(temp_dir, file.filename)

    try:
        with open(temp_file_path, 'wb') as temp_file:
            temp_file.write(await file.read())

        image_text_data = download_and_store(temp_file_path)

        dir = "/tmp/training_data"
        os.makedirs(dir, exist_ok=True)

        response_data = {"downloaded_data": []}

        for data in image_text_data:
            image_bytes = base64.b64decode(data.image_data)
            image_filename = os.path.join(dir, data.image_url.split('/')[-1])
            with open(image_filename, 'wb') as image_file:
                image_file.write(image_bytes)
            text_filename = os.path.join(dir, f"{data.image_url.split('/')[-1][:-4]}.txt")
            with open(text_filename, 'w') as text_file:
                text_file.write(data.text)

            response_data["downloaded_data"].append({
                "image_url": data.image_url,
                "image_filename": image_filename,
                "text_filename": text_filename
            })
        train_main(json_file_path)
        
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    uvicorn.run(app, port=8000, host="0.0.0.0")