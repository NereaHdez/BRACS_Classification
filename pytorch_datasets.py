import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from cv2 import imread
import pandas as pd

class Dataset(data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, inputs, labels, transform):
        'Initialization'
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)
        
    def __getitem__(self, index):
        'Generates one sample of data'
    
        file = self.inputs[index]
        x = imread(file).astype(np.uint8)

        if self.transform:
            x = self.transform(transforms.ToPILImage()(x))
        
        y = self.labels[index]
        
        y = torch.from_numpy(np.asarray(y)).float()
   
        return x, y, file


class TestDataset(data.Dataset):
    'Characterizes the test dataset'
    def __init__(self, inputs, labels, transform):
        'Initialization'
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
    
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)
        
    def __getitem__(self, index):
        'Generates one sample of data'
    
        files = self.inputs[index]
        print(files)
        images = []
        x = imread(files).astype(np.uint8)
        if self.transform:
            x = self.transform(transforms.ToPILImage()(x)) 
        images.append(x)
            
        y = self.labels[index]
        y = torch.from_numpy(np.asarray(y)).float()
   
        return images, y, files