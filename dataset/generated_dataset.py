import torch
import numpy as np, os, struct
from dataset.base_dataset import BaseDataset
from os.path import expanduser, join
from dataset.markable_dataset import MarkableDataset
from PIL import Image
import cv2

class GeneratedCifarDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None, transform=None):
        self.num_images = 20100
        
        if mode == 'val':
            assert val_frac is not None

        if path is None:
            self.path = '/data/zsl/generative-models/figs_cifar'
        else:
            self.path = path
        
        super(GeneratedCifarDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize,
            transform=transform
        )
        
        self.aux_data = {}
        
        # assert self.num_images % self.get_num_classes() == 0
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        # Initialize an empty NumPy array to store the images
        self.data = np.empty((self.num_images, 1024, 1024, 3), dtype=np.uint8)
        self.labels = np.empty((self.num_images, 1), dtype=np.uint8)

        # Loop through each image number and load it as a grayscale image
        for i in range(self.num_images):
            image_path = os.path.join(self.path, f"{i}.jpg")
            # image = Image.open(image_path).convert("L")  # "L" mode converts to grayscale
            image = Image.open(image_path)
            image = np.array(image)

            self.data[i, :, :, :] = image
            self.labels[i, 0] = i // (self.num_images // self.get_num_classes())

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10
    
    def update(self, i, aux_data=None):
        self.aux_data[i] = aux_data
        
class GeneratedCifarFinetuneDataset(GeneratedCifarDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path='/data/zsl/generative-models/figs_cifar_finetune', resize=None):
        self.num_images = 610*10
        
        if mode == 'val':
            assert val_frac is not None

        if path is None:
            self.path = '/data/zsl/generative-models/figs_cifar'
        else:
            self.path = path
        
        super(GeneratedCifarDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
        self.aux_data = {}
        
        # assert self.num_images % self.get_num_classes() == 0
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        # Initialize an empty NumPy array to store the images
        self.data = np.empty((self.num_images, 1024, 1024, 3), dtype=np.uint8)
        self.labels = np.empty((self.num_images, 1), dtype=np.uint8)

        # Loop through each image number and load it as a grayscale image
        num_cls = self.get_num_classes()
        cc = 0
        for cls_id in range(num_cls):
            for i in range(self.num_images//num_cls):
                image_path = os.path.join(self.path, f"{cls_id}_{i}.jpg")
                # image = Image.open(image_path).convert("L")  # "L" mode converts to grayscale
                image = Image.open(image_path)
                image = np.array(image)

                self.data[cc, :, :, :] = image
                self.labels[cc, 0] = cls_id
                cc += 1

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10
    
    def update(self, i, aux_data=None):
        self.aux_data[i] = aux_data

class GeneratedMnistDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        self.num_images = 1000
        
        if mode == 'val':
            assert val_frac is not None

        if path is None:
            home = expanduser("~")
            self.path = os.path.join('data', 'generated2')
        else:
            self.path = path
        
        super(GeneratedMnistDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):       
        # Initialize an empty NumPy array to store the images
        self.data = np.empty((self.num_images, 1024, 1024, 1), dtype=np.uint8)
        self.labels = np.empty((self.num_images, 1), dtype=np.uint8)

        # Loop through each image number and load it as a grayscale image
        for i in range(100):
            image_path = os.path.join(self.path, f"{i}.jpg")
            image = Image.open(image_path).convert("L")  # "L" mode converts to grayscale
            image = np.array(image)
            
            # 1. 增强对比度：使用直方图均衡化来增强对比度
            image = cv2.equalizeHist(image)

            # # 2. 噪声滤除：使用形态学操作（开运算）去除小的噪声
            # kernel = np.ones((25, 25), np.uint8)
            # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)

            # # 3. 连通性增强：使用形态学操作（膨胀和腐蚀）增强图像的连通性
            # kernel = np.ones((45, 45), np.uint8)
            # image = cv2.dilate(image, kernel, iterations=2)
            # image = cv2.erode(image, kernel, iterations=2)

            # 4. 抗锯齿处理：使用高斯滤波器减少锯齿效应
            image = cv2.GaussianBlur(image, (50, 50), 0)

            # 5. 缩放到28x28的大小
            image = cv2.resize(image, (28, 28))

            self.data[i, :, :, 0] = image
            self.labels[i, 0] = i // (self.num_images // self.get_num_classes())

        self.data = np.repeat(self.data, 100, axis=0)
        self.labels = np.repeat(self.labels, 100, axis=0)
        
        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 10
    
    def update(self, i, aux_data=None):
        self.aux_data[i] = aux_data
        

class GeneratedImagenetDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        self.num_images = 1000
        
        if mode == 'val':
            assert val_frac is not None

        if path is None:
            home = expanduser("~")
            self.path = '/data/zsl/generative-models/figs_imagenet'
        else:
            self.path = path
        
        super(GeneratedImagenetDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
        self.aux_data = {}
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):       
        # Initialize an empty NumPy array to store the images
        self.data = np.empty((self.num_images, 1024, 1024, 3), dtype=np.uint8)
        self.labels = np.empty((self.num_images, 1), dtype=np.uint8)

        # Loop through each image number and load it as a grayscale image
        for i in range(self.num_images):
            image_path = os.path.join(self.path, f"{i}.jpg")
            # image = Image.open(image_path).convert("L")  # "L" mode converts to grayscale
            image = Image.open(image_path)
            image = np.array(image)
            self.data[i, :, :, :] = image
            self.labels[i, 0] = i // (self.num_images // self.get_num_classes())

        # Perform splitting
        if val_frac is not None:
            self.partition_validation_set(mode, val_frac)
            
        self.labels = np.squeeze(self.labels)

    def get_num_classes(self):
        return 1000
    
    def update(self, i, aux_data=None):
        self.aux_data[i] = aux_data


class GeneratedCifarMarkableDataset(MarkableDataset, GeneratedCifarDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        GeneratedCifarDataset.__init__(self, normalize, mode, val_frac, normalize_channels, path, resize)
        MarkableDataset.__init__(self)
        
class GeneratedCifarFinetuneMarkableDataset(MarkableDataset, GeneratedCifarFinetuneDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path='/data/zsl/generative-models/figs_cifar_finetune+empty_prompt', resize=None):
        GeneratedCifarFinetuneDataset.__init__(self, normalize, mode, val_frac, normalize_channels, path, resize)
        MarkableDataset.__init__(self)
        
class GeneratedImagenetMarkableDataset(MarkableDataset, GeneratedImagenetDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        GeneratedImagenetDataset.__init__(self, normalize, mode, val_frac, normalize_channels, path, resize)
        MarkableDataset.__init__(self)
        
class GeneratedMnistMarkableDataset(MarkableDataset, GeneratedMnistDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        GeneratedMnistDataset.__init__(self, normalize, mode, val_frac, normalize_channels, path, resize)
        MarkableDataset.__init__(self)