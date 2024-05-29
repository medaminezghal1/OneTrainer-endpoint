import sys
import os
import csv
from urllib.request import urlopen
import base64
import tempfile
import shutil

sys.path.append('/home/OneTrainer-endpoint/scripts')
from train import main as train_main

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

def train_lora(csv_file_path, param):
    json_file_path = "/home/OneTrainer-endpoint/training_presets/sdxl_1.0_LoRA_style.json"
    temp_dir = tempfile.mkdtemp()

    try:
        image_text_data = download_and_store(csv_file_path)

        training_dir = "/tmp/training_data"
        os.makedirs(training_dir, exist_ok=True)

        for data in image_text_data:
            image_bytes = base64.b64decode(data.image_data)
            image_filename = os.path.join(training_dir, data.image_url.split('/')[-1])
            with open(image_filename, 'wb') as image_file:
                image_file.write(image_bytes)
            text_filename = os.path.join(training_dir, f"{data.image_url.split('/')[-1][:-4]}.txt")
            with open(text_filename, 'w') as text_file:
                text_file.write(data.text)

        # Execute the training process with the given JSON configuration file
        train_main(json_file_path)

    except Exception as e:
        raise RuntimeError(f"Error during training: {e}")
    finally:
        shutil.rmtree(temp_dir)

# Example usage:
# train_lora('/path/to/your/csvfile.csv', 'some_param')
