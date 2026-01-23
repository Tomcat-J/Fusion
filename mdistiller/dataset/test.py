import os
import sys
import json
import pickle
import random
import math
import numpy as np
from torchvision import transforms
import torch
from tqdm import tqdm
import re
from PIL import Image
from torch.utils.data import Dataset

from mdistiller.dataset.augmentations import apply_augment

def read_split_data(root: str, val_rate: float = 0.2):
        random.seed(6050)  # 保证随机结果可复现
        #0 1 2 7 312 1622 6050
        assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

        # 遍历文件夹，一个文件夹对应一个类别
        flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
        # 排序，保证各平台顺序一致
        flower_class.sort()
        # 生成类别名称以及对应的数字索引
        class_indices = dict((k, v) for v, k in enumerate(flower_class))
        json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)

        train_images_path = []  # 存储训练集的所有图片路径
        train_images_label = []  # 存储训练集图片对应索引信息
        val_images_path = []  # 存储验证集的所有图片路径
        val_images_label = []  # 存储验证集图片对应索引信息
        every_class_num = []  # 存储每个类别的样本总数
        supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
        # 遍历每个文件夹下的文件
        for cla in flower_class:
            cla_path = os.path.join(root, cla)
            # 遍历获取supported支持的所有文件路径
            images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
            # 排序，保证各平台顺序一致
            images.sort()
            # 获取该类别对应的索引
            image_class = class_indices[cla]
            # 记录该类别的样本数量
            every_class_num.append(len(images))
            # 按比例随机采样验证样本
            val_path = random.sample(images, k=int(len(images) * val_rate))

            for img_path in images:
                if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
                else:  # 否则存入训练集
                    train_images_path.append(img_path)
                    train_images_label.append(image_class)


        trimmed_paths = [path.replace('/home/data/pyl/lung/super/', '') for path in val_images_path]
        trimmed_further_paths = [re.sub(r'^[^/]+/', '', path) for path in trimmed_paths]
        final_trimmed_paths = [re.sub(r'_.*$', '', path) for path in trimmed_further_paths]
        filtered_count = sum(1 for path in final_trimmed_paths if len(path) == 4)

read_split_data("/home/data/pyl/lung/super")