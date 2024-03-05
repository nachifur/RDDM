import os
import shutil
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

data_dir = '/home/liu/disk12t/liu_data/dataset/CelebA/img_align_celeba'
train_dir = '/home/liu/disk12t/liu_data/dataset/CelebA/train/'
val_dir = '/home/liu/disk12t/liu_data/dataset/CelebA/val/'
test_dir = '/home/liu/disk12t/liu_data/dataset/CelebA/test/'

# Create the directories if they don't exist
for dir in [train_dir, val_dir, test_dir]:
    if not os.path.exists(dir):
        os.makedirs(dir)

data = np.loadtxt('/home/liu/disk12t/liu_data/dataset/CelebA/Eval/list_eval_partition.txt',dtype=str)

# Split image paths into train, val, and test sets
num_images = len(data)

def copy_file(file_data):
    file_name = file_data[0]
    flag = file_data[1]
    src_path = Path(data_dir)/file_name
    if flag=="0":
        save_path = Path(train_dir)/file_name
    elif flag=="1":
        save_path = Path(val_dir)/file_name
    elif flag=="2":
        save_path = Path(test_dir)/file_name
    shutil.copy(src_path,save_path)

if __name__ == '__main__':
    # Use all available CPU cores for processing
    pool = Pool(processes=10)
    results = []
    # Copy the image files to the appropriate directories in parallel
    for i in range(num_images):
        file_data = data[i]
        results.append(pool.apply_async(copy_file, args=(file_data,)))
        if i % 100 ==0:
            print("save "+str(i)+" -th")
    # Close the pool and wait for all processes to complete
    pool.close()
    pool.join()
