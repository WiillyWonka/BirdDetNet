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

import matplotlib.pyplot as plt

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
        self.threshold = 0.3
        self.additional_offset = 5

    def __len__(self):
        return len(self.img_paths)

    def create_labels(self, label):
        gt_keypoint_label = np.zeros((self.n_classes, self.img_size[0], self.img_size[1]))
        gt_reg_label = np.zeros((self.img_size[0], self.img_size[1], 3))
        for gt_l in label:
            center_y = int(self.img_size[0] - gt_l[0] / self.cell_size)
            center_x = int(self.img_size[1] / 2 - gt_l[1] / self.cell_size)
            gt_keypoint_label[int(gt_l[2])][center_y][center_x] = 1
            gt_reg_label[center_y][center_x][0] = gt_l[3]
            gt_reg_label[center_y][center_x][1] = gt_l[4]
            gt_reg_label[center_y][center_x][2] = gt_l[5]
        gt_keypoint_label = np.argmax(gt_keypoint_label, axis=0)
        return gt_keypoint_label, gt_reg_label

    def get_object_info(self, label):
        center_y = int(self.img_size[0] - label[0] / self.cell_size)
        center_x = int(self.img_size[1] / 2 - label[1] / self.cell_size)
        width = label[4] / self.cell_size
        height = label[5] / self.cell_size

        return center_x, center_y, width, height

    # rect = (x1a, y1a, x2a, y2a)
    def is_in_rect(self, rect, point):
        return point[0] >= rect[0] and point[1] >= rect[1] \
            and point[0] <= rect[2] and point[1] <= rect[3]

    def calc_intersect_percentage(self, candidate, rect):
        candidate_area = (candidate[0] - candidate[2]) * (candidate[1] - candidate[3])

        left = max(candidate[0], rect[0])
        top = max(candidate[1], rect[1])
        right = min(candidate[2], rect[2])
        bottom = min(candidate[3], rect[3])

        if left >= right or top >= bottom:
            return 0

        intersect_area = (right - left) * (bottom - top)

        return intersect_area / candidate_area

    def move_rect_along_axis(self, bound1, bound2, p, max):
        delta = 0
        if p < bound1:
            additional_offset = min(self.additional_offset, p)
            delta = bound1 - p + additional_offset
        elif p > bound2:
            additional_offset = min(self.additional_offset, max - p)
            delta = bound2 - p - additional_offset

        return bound1 - delta, bound2 - delta

    def fix_coords(self, x1, y1, x2, y2, labels):
        for label in labels:
            c_x, c_y, width, height = self.get_object_info(label)
            rect = (x1, y1, x2, y2)

            if self.is_in_rect(rect, (c_x, c_y)):
                continue
            
            left = c_x - width / 2
            right = c_x + width / 2
            top = c_y - height / 2
            down = c_y + height / 2

            candidate = (left, top, right, down)
            
            if (self.calc_intersect_percentage(candidate, rect) < self.threshold):
                continue

            x1, x2 = self.move_rect_along_axis(x1, x2, c_x, self.img_size[1] - 1)
            y1, y2 = self.move_rect_along_axis(y1, y2, c_y, self.img_size[0] - 1)

        return x1, y1, x2, y2


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
                elif i == 1:
                    x1a, y1a, x2a, y2a = xc, 0, self.img_size[1], yc
                elif i == 2:
                    x1a, y1a, x2a, y2a = 0, yc, xc, self.img_size[0]
                elif i == 3:
                    x1a, y1a, x2a, y2a = xc, yc, self.img_size[1], self.img_size[0]

                x1b, y1b, x2b, y2b = self.create_new_coords(x1a, y1a, x2a, y2a)
                x1b, y1b, x2b, y2b = self.fix_coords( \
                    x1b, y1b, x2b, y2b, self.labels[indices[i]])
                    
                image_new[y1a:y2a, x1a:x2a, :3] = image=image[int(y1b):int(y2b), int(x1b):int(x2b), :3]
                gt_keypoint_label_new[y1a:y2a, x1a:x2a] = label[0][int(y1b):int(y2b), int(x1b):int(x2b)]
                gt_reg_label_new[y1a:y2a, x1a:x2a, :3] = label[1][int(y1b):int(y2b), int(x1b):int(x2b)]

            return image_new, gt_keypoint_label_new, gt_reg_label_new