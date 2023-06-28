import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
from cv2 import imread
import pandas as pd
import slideflow as sf



class Dataset(data.Dataset):
    def __init__(self, inputs, labels, transform=None):
        """
        Inicialización del conjunto de datos.

        Args:
            inputs (list): Lista de rutas de archivos de entrada.
            labels (list): Lista de etiquetas correspondientes a los datos de entrada.
            transform (callable, optional): Transformaciones a aplicar a los datos. Por defecto es None.
        """
        self.labels = labels
        self.inputs = inputs
        self.transform = transform


    def __len__(self):
        """
        Devuelve la longitud del conjunto de datos.

        Returns:
            int: Número total de muestras en el conjunto de datos.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Genera una muestra de datos en función del índice.

        Args:
            index (int): Índice de la muestra.

        Returns:
            tuple: Tupla que contiene la imagen, la etiqueta y el archivo.
        """
        macenko = sf.norm.autoselect('macenko')
        file = self.inputs[index]
        x = imread(file).astype(np.uint8)
         # Aplicar normalización de tinción con Macenko
        x = macenko.transform(x)
        if self.transform:
            x = self.transform(transforms.ToPILImage()(x))
        
        y = self.labels[index]
        
        y = torch.from_numpy(np.asarray(y)).float()
   
        return x, y, file


class TestDataset(data.Dataset):
    def __init__(self, inputs, labels, transform=None):
        """
        Inicialización del conjunto de datos.

        Args:
            inputs (list): Lista de rutas de archivos de entrada.
            labels (list): Lista de etiquetas correspondientes a los datos de entrada.
            transform (callable, optional): Transformaciones a aplicar a los datos. Por defecto es None.
        """
        self.labels = labels
        self.inputs = inputs
        self.transform = transform


    def __len__(self):
        """
        Devuelve la longitud del conjunto de datos.

        Returns:
            int: Número total de muestras en el conjunto de datos.
        """
        return len(self.inputs)

    def __getitem__(self, index):
        """
        Genera una muestra de datos en función del índice.

        Args:
            index (int): Índice de la muestra.

        Returns:
            tuple: Tupla que contiene la imagen, la etiqueta y el archivo.
        """
        macenko = sf.norm.autoselect('macenko')
        file = self.inputs[index]
        x = imread(file).astype(np.uint8)
         # Aplicar normalización de tinción con Macenko
        x = macenko.transform(x)
        if self.transform:
            x = self.transform(transforms.ToPILImage()(x))
        
        y = self.labels[index]
        
        y = torch.from_numpy(np.asarray(y)).float()
   
        return x, y, file