import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from config import Config
from random import uniform
from random import random
from imageaug import crop_square, random_horizontal_flip

import xml.etree.ElementTree as ET


def my_collate_fn(batch):

    images = torch.stack(list(map(lambda x: torch.tensor(x[0]).double(), batch)))
    coordinates = list(map(lambda x: x[1], batch))
    pathes = list(map(lambda x: x[2], batch))

    return images, coordinates, pathes


def create_voc_datasets(voc_dataset_dir, split_ratio=0.2):
    annotations_dir = os.path.join(voc_dataset_dir, 'Annotations/All_Annotations/')
    imgnames_dir=os.path.join(voc_dataset_dir, 'ImageSets/All_train_imgnames.txt')
    imgnames = open(imgnames_dir, 'r')
    imgs = imgnames.readlines()
    imgnames.close()

    coordinates = []

    for img in imgs:
        annotation = img.strip() + '.xml'
        annotation_path = os.path.join(annotations_dir, annotation)


        def extract_info(annotation_path, classes=Config.VOC_CLASS):
            tree = ET.parse(annotation_path)
            root = tree.getroot()
            file_path = root.findall('filename')[0].text
            coordinates = []

            for i in root.findall('object'):
                if i.find('name').text == classes:
                    xmin = i.find('bndbox').find('xmin').text
                    ymin = i.find('bndbox').find('ymin').text
                    xmax = i.find('bndbox').find('xmax').text
                    ymax = i.find('bndbox').find('ymax').text

                    try:
                        coordinate = (int(ymin), int(xmin), int(ymax), int(xmax), 1)
                        coordinates.append(coordinate)
                    except:
                        continue
                else:
                    continue
            return (file_path, coordinates)

        item = extract_info(annotation_path)
        if len(item[1]) == 0:
            continue
        else:
            coordinates.append(item)

    train_annotation, val_annotation = train_test_split(coordinates, test_size=0.1)

    train_dataset = VOCDataset(
        os.path.join(voc_dataset_dir),
        train_annotation,
        image_size=Config.IMAGE_SIZE)

    validation_dataset = VOCDataset(
        os.path.join(voc_dataset_dir),
        val_annotation,
        image_size=Config.IMAGE_SIZE)

    return train_dataset, validation_dataset


class VOCDataset(Dataset):

    def __init__(self, images_dir, annotation, image_size=640, transform=None, random_flip=Config.RANDOM_FLIP, random_crop=True,
                 random_color_jitter=Config.RANDOM_COLOR_JITTER,
                 mode='train'):
        super().__init__()
        self.images_dir = os.path.join(images_dir, 'Data/train_and_validate/All')
        self.annotation = annotation
        self.transform = transform
        self.image_size = image_size
        self.random_flip = random_flip
        self.transform = None
        self.random_color_jitter = random_color_jitter
        self.random_crop = random_crop
        self.mode = mode

    def __image_loader(self, image_path):
        return cv2.imread(image_path)

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):
        file_path, coordinates = self.annotation[index]
        file_path = os.path.join(self.images_dir, file_path)
        image = self.__image_loader(file_path)
        #image = image - np.array([104, 117, 123], dtype=np.uint8)
        # if self.mode == 'train':
        #     if random() < 0.5:
        #         ratio = uniform(Config.MIN_CROPPED_RATIO, Config.MAX_CROPPED_RATIO)
        #     else:
        #         ratio = 1
        # image, coordinates = crop_square(
        #     image, coordinates, ratio, Config.KEEP_AREA_THRESHOLD)
        # image, coordinates = \
        #     random_horizontal_flip(image, coordinates)

        # scale coordinate
        height, width = image.shape[:2]
        width_scale, height_scale = 640.0 / width, 640.0 / height
        coordinates = np.array(list(map(lambda x: [
            x[0] * height_scale,
            x[1] * width_scale,
            x[2] * height_scale,
            x[3] * width_scale,
            *x[4:]
        ], coordinates)))
        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)

        return (image, coordinates, file_path)