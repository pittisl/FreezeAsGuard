import os
import random
import shutil
from tqdm import tqdm

random.seed(2024)

src_data_path = "FamousFaces-25"
target_data_path = "ff25"

items = os.listdir(src_data_path)
folders = [item for item in items if os.path.isdir(os.path.join(src_data_path, item))]
print(folders, len(folders))

for folder in tqdm(folders):
    src_files = os.listdir(os.path.join(src_data_path, folder))
    src_image_files = [file for file in src_files if file.endswith(('.png'))]
    random.shuffle(src_image_files)
    N = len(src_image_files)
    N_train = N // 2
    
    # extract train samples
    for i, filename in enumerate(src_image_files[:N_train]):
        new_filename = f"img_{i+1}.{filename.split('.')[-1]}"
        old_path = os.path.join(src_data_path, folder, filename)

        if not os.path.exists(os.path.join(target_data_path, "train", folder)):
            os.makedirs(os.path.join(target_data_path, "train", folder))

        new_path = os.path.join(target_data_path, "train", folder, new_filename)

        shutil.copyfile(old_path, new_path)

    # extract test samples
    for i, filename in enumerate(src_image_files[N_train:]):
        new_filename = f"img_{i+1}.{filename.split('.')[-1]}"
        old_path = os.path.join(src_data_path, folder, filename)

        if not os.path.exists(os.path.join(target_data_path, "test", folder)):
            os.makedirs(os.path.join(target_data_path, "test", folder))

        new_path = os.path.join(target_data_path, "test", folder, new_filename)

        shutil.copyfile(old_path, new_path)