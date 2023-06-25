"""Script to extract and save patches."""
from cv2 import imread, imwrite
from pathlib import Path
from tqdm import tqdm
import numpy as np 
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import numpy as np
from PIL import Image
import argparse

# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para la creación de patches')

parser.add_argument('--patch_size', type=int, default=512,
                    help='Tamaño del patch')

# Parsear los argumentos
args = parser.parse_args()

# Acceder a los valores de los argumentos
patch_size = args.patch_size
folder_name='_RoI_patches'+str(patch_size)

def detect_emborronamiento_fft(image, size=60, thresh=10, vis=False):
    # Tomamos las dimensiones de la imagen y determinamos el centro
    (h, w) = image.shape
    (cX, cY) = (int(w / 2.0), int(h / 2.0))
    img_float32 = np.float32(image)
    # Aplicamos la transformada de Fourier
    dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # Centramos
    # Anulamos un recuadro de tamaño size x size
    dft_shift[cY - size:cY + size, cX - size:cX + size] = 0
    dft_shift = np.fft.ifftshift(dft_shift)
    recon = cv2.idft(dft_shift)
    # Calculamos la magnitud de la imagen reconstruida
    # y la media (otra forma de obtener la magnitud)
    magnitude = 20 * np.log(cv2.magnitude(recon[:, :, 0], recon[:, :, 1]) + 1e-10)  # Agregamos una pequeña constante
    mean = np.mean(magnitude)
    # La imagen está emborronada si el valor medio de la magnitud es
    # menor que un umbral.
    return (mean, mean <= thresh)

datasets = ['train', 'test', 'val']
clases_roi = pd.Series(['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC'])

files_RoI = []

for i in datasets:
    paths_RoI = './BRACS_RoI/latest_version/' + i + '/' + clases_roi + '/'
    for j in range(7):
        aux = glob.glob(paths_RoI[j] + '*.png')
        files_RoI += aux

patch_size = patch_size
thr = 270

for filename in tqdm(files_RoI):
    f = str(filename)
    save_name = f.split('/')[-1].split('.')[0]
    save_path = f.split('_')[0] + folder_name + '/' + '/'.join(f.split('/')[3:-1])

    # Verificar si el directorio ya existe
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    im = Image.open(f)
    nobgr_img_blocks = []
    # Redimensionar la imagen si no cumple con el tamaño mínimo del parche
    if im.width < patch_size or im.height < patch_size:
        im = im.resize((patch_size, patch_size))
    for j in range(0, im.width, patch_size):
        for i in range(0, im.height, patch_size):
            block = np.array(im.crop((j, i, j + patch_size, i + patch_size)))
            if block.shape == (patch_size, patch_size, 3):
                gray_block = np.array(Image.fromarray(block).convert('L'))  # Convertir a escala de grises
                # Aplicar nuestro detector de desenfoque utilizando FFT
                (mean, blurry) = detect_emborronamiento_fft(gray_block, size=60, thresh=thr)
                if (
                    np.mean(block[:, :, 0]) < 220.0
                    and np.mean(block[:, :, 1]) < 220.0
                    and np.mean(block[:, :, 2]) < 220.0
                    and np.min(gray_block) > 20
                ):
                    nobgr_img_blocks.append(block)

   
    if len(nobgr_img_blocks) < 1:
        block = np.array(im.resize((patch_size, patch_size)))
        nobgr_img_blocks.append(block)
    for index, block in enumerate(nobgr_img_blocks):
        save_filename = f'{save_path}/{save_name}_{index}.jpeg'
        if not os.path.isfile(save_filename):
            Image.fromarray(block).save(save_filename)
        else:
            print('already converted')