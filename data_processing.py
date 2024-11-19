import os
import shutil

import pandas as pd
from tqdm import tqdm

# 数据集根目录
root = r"/ssd/src_data/Manually_Annotated_Images/"
# training.csv
training_csv = r"/ssd/src_data/training.csv"
# validation.csv
validation_csv = r"/ssd/src_data/validation.csv"

dst_train = r"/ssd/src_data/new_datasets_ImageFolder/train"
os.makedirs(dst_train, exist_ok=True)
dst_val = r"/ssd/src_data/new_datasets_ImageFolder/val"
os.makedirs(dst_val, exist_ok=True)

# subDirectory_filePath列是图片的路径，expression列是标签
# 用标签创建子文件夹，然后对应图片移动到对应文件夹
df_train = pd.read_csv(training_csv)
df_train = df_train[["subDirectory_filePath", "expression"]]

for i in tqdm(range(len(df_train))):
    img_path = os.path.join(root, df_train.iloc[i, 0])
    label = df_train.iloc[i, 1]
    dst = os.path.join(dst_train, str(label))
    os.makedirs(dst, exist_ok=True)
    if os.path.exists(img_path):
        shutil.copy(img_path, dst)

df_val = pd.read_csv(validation_csv)
df_val = df_val[["subDirectory_filePath", "expression"]]

for i in tqdm(range(len(df_val))):
    img_path = os.path.join(root, df_val.iloc[i, 0])
    label = df_val.iloc[i, 1]
    dst = os.path.join(dst_val, str(label))
    os.makedirs(dst, exist_ok=True)
    if os.path.exists(img_path):
        shutil.copy(img_path, dst)

# 0到7的类别标签保留，其余的都转为一个标签8
dst_train_8 = r"/ssd/src_data/new_datasets_ImageFolder/train/8"
dst_train_9 = r"/ssd/src_data/new_datasets_ImageFolder/train/9"
dst_train_10 = r"/ssd/src_data/new_datasets_ImageFolder/train/10"
# 将9和10类别的图片移动到8类别文件夹，然后删除9和10文件夹
files_9 = os.listdir(dst_train_9)
files_10 = os.listdir(dst_train_10)
for file in files_9:
    shutil.move(os.path.join(dst_train_9, file), os.path.join(dst_train_8, file))
for file in files_10:
    shutil.move(os.path.join(dst_train_10, file), os.path.join(dst_train_8, file))
shutil.rmtree(dst_train_9)
shutil.rmtree(dst_train_10)

dst_val_8 = r"/ssd/src_data/new_datasets_ImageFolder/val/8"
dst_val_9 = r"/ssd/src_data/new_datasets_ImageFolder/val/9"
dst_val_10 = r"/ssd/src_data/new_datasets_ImageFolder/val/10"
# 将9和10类别的图片移动到8类别文件夹，然后删除9和10文件夹
files_9 = os.listdir(dst_val_9)
files_10 = os.listdir(dst_val_10)
for file in files_9:
    shutil.move(os.path.join(dst_val_9, file), os.path.join(dst_val_8, file))
for file in files_10:
    shutil.move(os.path.join(dst_val_10, file), os.path.join(dst_val_8, file))
shutil.rmtree(dst_val_9)
shutil.rmtree(dst_val_10)

# class_to_idx = {
#     'Neutral': 0,
#     'Happiness': 1,
#     'Sadness': 2,
#     'Surprise': 3,
#     'Fear': 4,
#     'Disgust': 5,
#     'Anger': 6,
#     'Contempt': 7,
#     # 可以继续添加其他目录和对应的标签
# }

