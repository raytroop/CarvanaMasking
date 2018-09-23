import os
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_path = os.path.abspath(os.path.join(dir_path, os.pardir))

data_dir = os.path.join(parent_path, 'data')
train_dir = os.path.join(data_dir, 'train')
train_masks_dir = os.path.join(data_dir, 'train_masks')

metadata_csv = os.path.join(data_dir, 'metadata.csv')
train_masks_csv = os.path.join(data_dir, 'train_masks.csv')

metadata_df = pd.read_csv(metadata_csv)
train_masks_df = pd.read_csv(train_masks_csv)

imgs_name = train_masks_df['img']

def read_img(img_name):
    img_path = os.path.join(train_dir, img_name)
    img = io.imread(img_path)
    img = img.astype(np.float32) / 255.0
    return img
def read_mask(mask_name):
    mask_path = os.path.join(train_masks_dir, mask_name)
    mask = io.imread(mask_path)
    mask = mask.astype(np.float32) / 255.0
    return mask
def img2mask_name(img_name):
    return img_name[:-4] + '_mask.gif'

img_i = imgs_name[4500]
img = read_img(img_i)
mask = read_mask(img2mask_name(img_i))
print(img.shape)
print(mask.shape)
print(np.unique(mask))
plt.imshow(img)
plt.imshow(mask, alpha=0.5)
plt.show()
print(img.shape)
print(len(imgs_name))
