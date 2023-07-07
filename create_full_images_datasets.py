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
from sklearn import preprocessing

clases = pd.Series(['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC'])
datasets = ['train', 'test', 'val']
clases_roi = pd.Series(['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC'])


datasets=['train','test','val']
paths={}


# 3 Clases
clases3 = ['AT', 'BT', 'MT']
AT = ['FEA', 'ADH']
BT = ['N', 'PB', 'UDH']
MT = ['DCIS', 'IC']

ohe = preprocessing.OneHotEncoder(sparse=False)
classes = np.array(clases3)
ohe.fit(classes.reshape(-1, 1))

data_RoI = {}
data_RoI['train'] = {'x': [], 'y': []}
data_RoI['val'] = {'x': [], 'y': []}
data_RoI['test'] = {'x': [], 'y': []}

for i in datasets:
    files_RoI =list()
    paths_RoI = './BRACS_RoI/latest_version/' + i + '/' + clases_roi + '/'
    for j in range(7):
        aux = glob.glob(paths_RoI[j] + '*.png')
        files_RoI += aux
        data_RoI[i]['x'].extend(aux)

    
    label = [file.split('/')[-2].split('_')[-1] for file in files_RoI]
    label_mapping = {'AT': AT, 'BT': BT, 'MT': MT}
    label = [next(key for key, value in label_mapping.items() if elemento in value) for elemento in label]

    data_RoI[i]['y'].extend(label)
    data_RoI[i]['x'] = np.asarray(data_RoI[i]['x'])
    data_RoI[i]['y'] = np.asarray(data_RoI[i]['y'])
    data_RoI[i]['y'] = ohe.transform(data_RoI[i]['y'].reshape(-1, 1))

import pickle
with open('data_RoI_full.pkl', 'wb') as fp:
    pickle.dump(data_RoI, fp)
    print('dictionary saved successfully to file')