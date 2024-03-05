import multiprocessing
import os

from PIL import Image

# 定义输入输出文件夹路径
input_folder = '/home/liu/disk12t/liu_data/dataset/CelebA/test'
output_folder = '/home/liu/disk12t/liu_data/dataset/CelebA/img_align_celeba_test_crop_256x256'

img_size = 256

# 如果输出文件夹不存在，则创建这个文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义缩放和保存图像的函数
def resize_and_save(filepath):
    with Image.open(filepath) as im:
        # 缩放图像
        cx = 89
        cy = 121
        x1 = cy - img_size
        x2 = cy + img_size
        y1 = cx - img_size
        y2 = cx + img_size
        im = im.crop((y1, x1, y2, x2))
        size = (img_size, img_size)
        im_resized = im.resize(size)
        # 构造输出文件路径
        filename = os.path.basename(filepath)
        output_filepath = os.path.join(output_folder, filename)
        # 保存缩放后的图像
        im_resized.save(output_filepath)

# 获取输入文件夹内所有图像文件的路径列表
files = os.listdir(input_folder)
image_files = [f for f in files if f.endswith('.jpg') or f.endswith('.png')]

# 指定使用多少个CPU核来处理图像文件
num_processes = 20
with multiprocessing.Pool(processes=num_processes) as pool:
    pool.map(resize_and_save, [os.path.join(input_folder, f) for f in image_files])
