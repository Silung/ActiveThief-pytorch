import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from dataset.base_dataset import BaseDataset
from dataset.markable_dataset import MarkableDataset

class GtsrbDataset(BaseDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, path=None, resize=None):
        if mode == 'val':
            assert val_frac is not None
            
        if path is None:
            self.path = os.path.join('data', 'GTSRB')
        else:
            self.path = path
        
        super(GtsrbDataset, self).__init__(
            normalize=normalize,
            mode=mode,
            val_frac=val_frac,
            normalize_channels=normalize_channels,
            resize=resize
        )
        
    def is_multilabel(self):
        return False

    def load_data(self, mode, val_frac):
        xs = []
        ys = []
        if mode != 'test' :
            for class_id in range(self.get_num_classes()):
                class_name = '{:05d}'.format(class_id)
                annotation_file = os.path.join(self.path, 'Final_Training', 'Images', class_name, f"GT-{class_name}.csv")
                annotations = pd.read_csv(annotation_file, sep=';')
                
                for index, row in annotations.iterrows():
                    filename = os.path.join(self.path, 'Final_Training', 'Images', class_name, row['Filename'])
                    width, height = row['Width'], row['Height']
                    roi_x1, roi_y1, roi_x2, roi_y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
                    class_id = row['ClassId']
                    xs.append({
                        'image_path': filename,
                        'width': width,
                        'height': height,
                        'roi': (roi_x1, roi_y1, roi_x2, roi_y2)
                    })
                    ys.append(class_id)
        else:
            annotation_file = os.path.join(self.path, "GT-final_test.csv")
            annotations = pd.read_csv(annotation_file, sep=';')
            
            for index, row in annotations.iterrows():
                filename = os.path.join(self.path, 'Final_Test', 'Images', row['Filename'])
                width, height = row['Width'], row['Height']
                roi_x1, roi_y1, roi_x2, roi_y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
                class_id = row['ClassId']
                xs.append({
                    'image_path': filename,
                    'width': width,
                    'height': height,
                    'roi': (roi_x1, roi_y1, roi_x2, roi_y2)
                })
                ys.append(class_id)
        
        images = []
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32))
        ])
        for item in xs:
            image = Image.open(item['image_path'])
            roi = item['roi']
            image = transform(image.crop(roi))  # Crop the image based on ROI
            image = np.array(image)
            images.append(image)
            
        if len(xs) == 1:
            self.data   = images[0]
            self.labels = ys[0]
        else:
            self.data   = np.stack(images)
            self.labels = np.array(ys)

    def get_num_classes(self):
        return 43

class GtsrbMarkableDataset(MarkableDataset, GtsrbDataset):
    def __init__(self, normalize=True, mode='train', val_frac=0.2, normalize_channels=False, resize=None):
        GtsrbDataset.__init__(self, normalize=normalize, mode=mode, val_frac=val_frac, normalize_channels=normalize_channels, resize=resize)
        MarkableDataset.__init__(self)
