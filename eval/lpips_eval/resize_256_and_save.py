import os
import sys

from PIL import Image

def resize_256_and_save(load_folder,save_folder):
    for filename in os.listdir(load_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(load_folder, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize((256, 256),resample=Image.Resampling.BICUBIC)  
            save_path = os.path.join(save_folder, filename)
            img.save(save_path)
            print("save:"+save_path)

if __name__ == '__main__':
    load_folder = sys.argv[1]
    save_folder = sys.argv[2]
    resize_256_and_save(load_folder,save_folder)
