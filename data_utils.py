import json
import os
from os import listdir
from os.path import join

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize,Grayscale
from imagecrop import FusionRandomCrop
from torchvision.transforms import functional as F



class TrainDataset(Dataset):
    def __init__(self, data_path, exp_path, patch_w=560, patch_h=315, rho=8, WIDTH=640, HEIGHT=360):

        self.imgs = open(data_path, 'r').readlines()
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.rho = rho
        self.train_path = os.path.join(exp_path, 'dataset/Train/')

    def __getitem__(self, index):

        value = self.imgs[index]
        img_names = value.split(' ')
        # 遍历文件夹，一个文件夹对应一个类别
        flower_class = [cla for cla in os.listdir(self.train_path) if os.path.isdir(os.path.join(self.train_path, cla))]
        # 排序，保证各平台顺序一致
        flower_class.sort()
        # 生成类别名称以及对应的数字索引
        class_indices = dict((k, v) for v, k in enumerate(flower_class))
        json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
        with open('class_indices.json', 'w') as json_file:
            json_file.write(json_str)
        # 获取该类别对应的索引
        images_label = class_indices[img_names[0].split('/')[0]]
        imgs=[]
        for i in range(len(img_names)-1):
            if img_names[i].split('/')[-1]!='None':
                img=cv2.imread(self.train_path + img_names[i])
                height, width = img.shape[:2]
                if height != self.HEIGHT or width != self.WIDTH:
                    img = cv2.resize(img, (self.WIDTH, self.HEIGHT))

                img = (img - self.mean_I) / self.std_I
                # img = np.mean(img, axis=2, keepdims=True)
                img = np.transpose(img, [2, 0, 1])
                imgs.append(img)
            else:
                imgs.append(np.zeros((3, self.WIDTH, self.HEIGHT)))
        microscope_num=int(img_names[-1][:-1])
        imgs = np.concatenate(imgs, axis=0)

        x = np.random.randint(self.rho, self.WIDTH - self.rho - self.patch_w)
        y = np.random.randint(self.rho, self.HEIGHT - self.rho - self.patch_h)

        input_tesnor = imgs[:, y: y + self.patch_h, x: x + self.patch_w]

        org_img = torch.tensor(input_tesnor)
        images_label = torch.tensor(images_label)
        microscope_num=torch.tensor(microscope_num)


        return (org_img, images_label,microscope_num)

    def __len__(self):

        return len(self.imgs)


class ValDataset(Dataset):
    def __init__(self, data_path, patch_w=560, patch_h=315, rho=16, WIDTH=640, HEIGHT=360):
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.rho = rho

        self.work_dir = os.path.join(data_path, 'dataset')
        self.pair_list = list(open(os.path.join(self.work_dir, 'Val_List.txt')))
        print(len(self.pair_list))
        self.img_path = os.path.join(self.work_dir, 'Val/')

    def __getitem__(self, index):

        img_pair = self.pair_list[index]
        pari_id = img_pair.split(' ')

        # 遍历文件夹，一个文件夹对应一个类别
        flower_class = [cla for cla in os.listdir(self.img_path) if os.path.isdir(os.path.join(self.img_path, cla))]
        # 排序，保证各平台顺序一致
        flower_class.sort()
        # 生成类别名称以及对应的数字索引
        class_indices = dict((k, v) for v, k in enumerate(flower_class))
        # 获取该类别对应的索引
        images_label = class_indices[pari_id[0].split('/')[0]]
        imgs = []
        for i in range(len(pari_id)-1):
            if pari_id[i].split('/')[-1]!='None':
                img=cv2.imread(self.img_path + pari_id[i])
                height, width = img.shape[:2]
                if height != self.HEIGHT or width != self.WIDTH:
                    img = cv2.resize(img, (self.WIDTH, self.HEIGHT))

                img = (img - self.mean_I) / self.std_I
                # img = np.mean(img, axis=2, keepdims=True)
                img = np.transpose(img, [2, 0, 1])
                imgs.append(img)
            else:
                imgs.append(np.zeros((3, self.WIDTH, self.HEIGHT)))

        microscope_num=int(pari_id[-1][:-1])
        imgs = np.concatenate(imgs, axis=0)

        x = np.random.randint(self.rho, self.WIDTH - self.rho - self.patch_w)
        y = np.random.randint(self.rho, self.HEIGHT - self.rho - self.patch_h)

        input_tesnor = imgs[:, y: y + self.patch_h, x: x + self.patch_w]

        org_img = torch.tensor(input_tesnor)
        images_label = torch.tensor(images_label)
        microscope_num=torch.tensor(microscope_num)


        return (org_img, images_label,microscope_num)



    def __len__(self):

        return len(self.pair_list)


class TestDataset(Dataset):
    def __init__(self, data_path, dataset_txt, patch_w=560, patch_h=315, rho=16, WIDTH=640, HEIGHT=360):
        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.WIDTH = WIDTH
        self.HEIGHT = HEIGHT
        self.rho = rho

        self.work_dir = os.path.join(data_path, 'dataset')
        self.pair_list = list(open(os.path.join(self.work_dir, dataset_txt)))
        print(len(self.pair_list))
        self.img_path = os.path.join(self.work_dir, 'Test/')

    def __getitem__(self, index):

        img_pair = self.pair_list[index]
        pari_id = img_pair.split(' ')
        # 遍历文件夹，一个文件夹对应一个类别
        flower_class = [cla for cla in os.listdir(self.img_path) if os.path.isdir(os.path.join(self.img_path, cla))]
        # 排序，保证各平台顺序一致
        flower_class.sort()
        # 生成类别名称以及对应的数字索引
        class_indices = dict((k, v) for v, k in enumerate(flower_class))
        # 获取该类别对应的索引
        images_label = class_indices[pari_id[0].split('/')[0]]
        imgs = []
        for i in range(len(pari_id) - 1):
            if pari_id[i].split('/')[-1] != 'None':
                img = cv2.imread(self.img_path + pari_id[i])
                height, width = img.shape[:2]
                if height != self.HEIGHT or width != self.WIDTH:
                    img = cv2.resize(img, (self.WIDTH, self.HEIGHT))

                img = (img - self.mean_I) / self.std_I
                # img = np.mean(img, axis=2, keepdims=True)
                img = np.transpose(img, [2, 0, 1])
                imgs.append(img)
            else:
                imgs.append(np.zeros((3, self.WIDTH, self.HEIGHT)))

        microscope_num = int(pari_id[-1][:-1])
        imgs = np.concatenate(imgs, axis=0)

        x = np.random.randint(self.rho, self.WIDTH - self.rho - self.patch_w)
        y = np.random.randint(self.rho, self.HEIGHT - self.rho - self.patch_h)

        input_tesnor = imgs[:, y: y + self.patch_h, x: x + self.patch_w]

        org_img = torch.tensor(input_tesnor)
        images_label = torch.tensor(images_label)
        microscope_num = torch.tensor(microscope_num)

        return (org_img, images_label, microscope_num)

    def __len__(self):

        return len(self.pair_list)
