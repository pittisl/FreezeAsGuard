import os
import csv
from tqdm import tqdm
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# by default `from_pretrained` loads the weights in float32
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.bfloat16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


split = "train"
fields = ["file_name", "text", "name"]
src_data_path = f"ff25/{split}"
csv_file_name = os.path.join(src_data_path, "metadata.csv")

metadata_dicts = []

items = os.listdir(src_data_path)
folders = [item for item in items if os.path.isdir(os.path.join(src_data_path, item))]

for folder in tqdm(folders):
    src_files = os.listdir(os.path.join(src_data_path, folder))
    src_image_files = [file for file in src_files if file.endswith(('.png'))]
    
    for filename in src_image_files:
        image_path = src_data_path + '/' + os.path.join(folder, filename)
        image = Image.open(image_path).convert('RGB')
        prompt = f"a photo of {folder.replace('_', ' ')} which shows"
        # prompt = f"Describe the photo of {folder.replace('_', ' ')} in detail: "
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.bfloat16)
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        text = (prompt + ' ' + generated_text).strip()
        # text = generated_text.strip()
        
        metadata_dicts.append({
            'file_name': os.path.join(folder, filename),
            'text': text,
            'name': folder.replace("_", " "),
        })

with open(csv_file_name, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
 
    # writing headers (field names)
    writer.writeheader()
 
    # writing data rows
    writer.writerows(metadata_dicts)