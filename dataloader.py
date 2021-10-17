import os
import glob
import numpy as np
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset,DataLoader
import torch
import random
from torchvision import transforms


class DataProcessor(Dataset):
    def __init__(self, x, y, size, n_classes, cell_size):
        self.img_paths = x
        self.labels = y
        self.img_size = size
        self.n_classes = n_classes
        self.img_center = (self.img_size[0] // 2, self.img_size[1] // 2)
        self.indices = [i for i in range(self.__len__())]
        self.mosaic_p = 0.1
        self.cell_size = cell_size
        self.augs = self.get_augmentation_pipeline()

    def __len__(self):
        return len(self.img_paths)

    def get_augmentation_pipeline(self):
        alb_transforms = A.Compose([
            A.Flip(p=0.7),
            A.ShiftScaleRotate(shift_limit=(0, 0), scale_limit=(0.5, 1.5), rotate_limit=(0, 0), p=0.6),
        ])
        return alb_transforms

    def create_labels(self, label):
        gt_keypoint_label = np.zeros((self.n_classes, self.img_size[0], self.img_size[1]))
        gt_reg_label = np.zeros((3, self.img_size[0], self.img_size[1]))
        for gt_l in label:
            center_y = int(self.img_size[0] - gt_l[0] / self.cell_size)
            center_x = int(self.img_size[1] / 2 - gt_l[1] / self.cell_size)
            gt_keypoint_label[int(gt_l[2])][center_y][center_x] = 1
            gt_reg_label[center_y][center_x][0] = gt_l[3]
            gt_reg_label[center_y][center_x][1] = gt_l[4]
            gt_reg_label[center_y][center_x][2] = gt_l[5]
        gt_keypoint_label = np.argmax(gt_keypoint_label, axis=0)
        return gt_keypoint_label, gt_reg_label

    def create_new_coords(self,x1a, y1a, x2a, y2a):
        delta_x = random.uniform(0, 200)
        delta_y = random.uniform(0, 200)
        lenght_x = x2a - x1a
        lenght_y = y2a - y1a
        if x1a==y1a:
            x2b = min(x2a + delta_x, self.img_size[1])
            y2b = min(y2a + delta_y, self.img_size[0])
            x1b = x2b - lenght_x
            y1b = y2b - lenght_y
        if x2a == self.img_size[1] and y1a==0:
            x1b = max(x1a - delta_x, 0)
            y2b = min(y2a + delta_y, self.img_size[0])
            x2b = x1b + lenght_x
            y1b = y2b - lenght_y
        if y2a ==  self.img_size[0] and x1a == 0:
            x2b = min(x2a + delta_x, self.img_size[1])
            x1b = x2b - lenght_x
            y1b = max(y1a - delta_y, 0)
            y2b = y1b + lenght_y
        if x2a == self.img_size[1] and y2a  ==  self.img_size[0]:
            x1b = max(x1a - delta_x, 0)
            y1b = max(y1a - delta_y, 0)
            x2b = x1b + lenght_x
            y2b = y1b + lenght_y
        return x1b,y1b,x2b,y2b

    def __getitem__(self, index):
        if random.random() > self.mosaic_p:
            xc = int(random.uniform(int(self.img_center[1] // 2), self.img_size[1]))
            yc = int(random.uniform(int(self.img_center[0] // 2), self.img_size[0]))
            indices = [index] + random.choices(self.indices, k=3)
            images = [cv2.imread(self.img_paths[i]) for i in indices]
            labels = [self.create_labels(self.labels[i]) for i in indices]
            gt_keypoint_label_new = np.zeros((self.img_size[0], self.img_size[1]))
            gt_reg_label_new = np.zeros((self.img_size[0], self.img_size[1],3))
            image_new = np.full((self.img_size[0], self.img_size[1], 3), 114, dtype=np.uint8)
            for i, (image, label) in enumerate(zip(images, labels)):
                if i == 0:
                    x1a, y1a, x2a, y2a = 0, 0, xc, yc
                    x1b,y1b,x2b,y2b = self.create_new_coords(x1a, y1a, x2a, y2a)
                    sample = self.augs(image=image[int(y1b):int(y2b), int(x1b):int(x2b), :3],
                                       masks=[label[0][int(y1b):int(y2b), int(x1b):int(x2b)],
                                              label[1][int(y1b):int(y2b), int(x1b):int(x2b)]])
                    image_new[y1a:y2a, x1a:x2a, :3] = sample["image"]
                    gt_keypoint_label_new[y1a:y2a, x1a:x2a] = sample["masks"][0]
                    gt_reg_label_new[y1a:y2a, x1a:x2a, :3] = sample["masks"][1]

                elif i == 1:
                    x1a, y1a, x2a, y2a = xc, 0, self.img_size[1], yc
                    x1b, y1b, x2b, y2b = self.create_new_coords(x1a, y1a, x2a, y2a)
                    sample = self.augs(image=image[int(y1b):int(y2b), int(x1b):int(x2b), :3],
                                       masks=[label[0][int(y1b):int(y2b), int(x1b):int(x2b)],
                                              label[1][int(y1b):int(y2b), int(x1b):int(x2b)]])
                    image_new[y1a:y2a, x1a:x2a, :3] = sample["image"]
                    gt_keypoint_label_new[y1a:y2a, x1a:x2a] = sample["masks"][0]
                    gt_reg_label_new[y1a:y2a, x1a:x2a, :3] = sample["masks"][1]

                elif i == 2:
                    x1a, y1a, x2a, y2a = 0, yc, xc, self.img_size[0]
                    x1b, y1b, x2b, y2b = self.create_new_coords(x1a, y1a, x2a, y2a)
                    sample = self.augs(image=image[int(y1b):int(y2b), int(x1b):int(x2b), :3],
                                       masks=[label[0][int(y1b):int(y2b), int(x1b):int(x2b)],
                                              label[1][int(y1b):int(y2b), int(x1b):int(x2b)]])
                    image_new[y1a:y2a, x1a:x2a, :3] = sample["image"]
                    gt_keypoint_label_new[y1a:y2a, x1a:x2a] = sample["masks"][0]
                    gt_reg_label_new[y1a:y2a, x1a:x2a, :3] = sample["masks"][1]

                elif i == 3:
                    x1a, y1a, x2a, y2a = xc, yc, self.img_size[1], self.img_size[0]
                    x1b, y1b, x2b, y2b = self.create_new_coords(x1a, y1a, x2a, y2a)
                    sample = self.augs(image=image[int(y1b):int(y2b), int(x1b):int(x2b), :3],
                                       masks=[label[0][int(y1b):int(y2b), int(x1b):int(x2b)],
                                              label[1][int(y1b):int(y2b), int(x1b):int(x2b)]])
                    image_new[y1a:y2a, x1a:x2a, :3] = sample["image"]
                    gt_keypoint_label_new[y1a:y2a, x1a:x2a] = sample["masks"][0]
                    gt_reg_label_new[y1a:y2a, x1a:x2a, :3] = sample["masks"][1]

            return image_new, gt_keypoint_label_new , gt_reg_label_new