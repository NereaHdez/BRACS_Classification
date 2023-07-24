'''create datasets'''
import numpy as np 
import os
import glob
import pandas as pd
import numpy as np
import os
import glob
import numpy as np
from sklearn import preprocessing
import numpy as np
from sklearn import preprocessing
import pandas as pd
import pickle
import os
import argparse

# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para la creación de patches')

parser.add_argument('--patch_size', type=int, default=512,
                    help='Tamaño de los patches')
parser.add_argument('--folder_patches', type=str, default='BRACS_RoI_patches512',
                    help='Carpeta de los patches')
parser.add_argument('--name_pkl', type=str, default='data_RoI512',
                    help='nombre archivo donde guardar datasets')
parser.add_argument('--n_clases', type=int, default=3,
                    help='Número de clases')
parser.add_argument('--full', type=int, default=0,
                    help='Indicador booleano para habilitar o deshabilitar tratar con full imagenes')
parser.add_argument('--wsi', type=int, default=0,
                    help='Indicador booleano para habilitar o deshabilitar el uso de imagenes WSI')
parser.add_argument('--patch_folder_WSI', type=str, default='_patches',
                    help='terminacion de la carpeta de los parches de las WSI')

args = parser.parse_args()

# Acceder a los valores de los argumentos
patch_size=args.patch_size
folder_patches = args.folder_patches
name_pkl=args.name_pkl
n_clases=args.n_clases
full=bool(args.full)
wsi=bool(args.wsi)
clases = pd.Series(['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC'])
datasets = ['train', 'test', 'val']
clases_roi = pd.Series(['0_N', '1_PB', '2_UDH', '3_FEA', '4_ADH', '5_DCIS', '6_IC'])
patch_folder_WSI=args.patch_folder_WSI
# 3 Clases
clases3 = ['AT', 'BT', 'MT']
AT = ['FEA', 'ADH']
BT = ['N', 'PB', 'UDH']
MT = ['DCIS', 'IC']

ohe = preprocessing.OneHotEncoder(sparse=False)

if n_clases==3:
    classes = np.array(clases3)
    ohe.fit(classes.reshape(-1, 1))
else:
    classes = np.array(clases)
    ohe.fit(classes.reshape(-1, 1))

data_RoI = {}
data_RoI['train'] = {'x': [], 'y': []}
data_RoI['val'] = {'x': [], 'y': []}
data_RoI['test'] = {'x': [], 'y': []}

for i in datasets:
    files_RoI = []
    files_WSI = []
    paths_RoI_pat = './'+folder_patches+'/' + i + '/' + clases_roi + '/'
    paths_RoI = './BRACS_RoI/latest_version/' + i + '/' + clases_roi + '/'


    if full:
        for j in range(7):
            aux = glob.glob(paths_RoI[j] + '*.png')
            files_RoI += aux
            data_RoI[i]['x'].extend(aux)  
    else:
        for j in range(7):
            path = paths_RoI_pat[j]
            aux = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.jpeg')]
            files_RoI.extend(aux)
            data_RoI[i]['x'].extend(aux)
    
    label = [file.split('/')[-2].split('_')[-1] for file in files_RoI]
    if n_clases==3:
        label_mapping = {'AT': AT, 'BT': BT, 'MT': MT}
        label = [next(key for key, value in label_mapping.items() if elemento in value) for elemento in label]

    
    if i=='train' and wsi:
        # Ruta del archivo Excel
        excel_file = "BRACS.xlsx"

        # Leer el archivo Excel
        df = pd.read_excel(excel_file)
        df_train=df[df['Set']=='Training']
        AT = ['FEA', 'ADH']
        BT = ['N', 'PB', 'UDH']
        MT = ['DCIS', 'IC']
        label_mapping = {'AT': AT, 'BT': BT, 'MT': MT}
        df_train['group'] = [next(key for key, value in label_mapping.items() if elemento in value) for elemento in df_train['WSI label']]
        
        def concatenar(row,texto=''):
            return 'BRACS_WSI'+texto+'/Group_'+ row['group'] + '/Type_' + row['WSI label']+'/'+row['WSI Filename']+'/'

        df_train['path_patch'] = df_train.apply(lambda row: concatenar(row, patch_folder_WSI), axis=1)
        paths_WSI_pat = list(np.unique(df_train['path_patch']))

        for j in range(len(paths_WSI_pat)):
            
            path = paths_WSI_pat[j]
            if not os.path.isdir(path):
                os.makedirs(path)
            aux = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.jpeg')]
            print(aux)
            files_WSI.extend(aux)
            data_RoI[i]['x'].extend(aux)

        if n_clases==3:
            label_WSI =[file.split('/')[-4].split('_')[-1] for file in files_WSI]
        else:
            label_WSI =[file.split('/')[-3].split('_')[-1] for file in files_WSI]

        label.extend(label_WSI)

    data_RoI[i]['y'].extend(label)
    data_RoI[i]['x'] = np.asarray(data_RoI[i]['x'])
    data_RoI[i]['y'] = np.asarray(data_RoI[i]['y'])
    data_RoI[i]['y'] = ohe.transform(data_RoI[i]['y'].reshape(-1, 1))

name_npy=name_pkl+'.npy'
name_pkl=name_pkl+'.pkl'

np.save(name_npy, data_RoI)

with open(name_pkl, 'wb') as fp:
    pickle.dump(data_RoI, fp)
    print('dictionary saved successfully to file')

df=pd.DataFrame()
df['x']=data_RoI['train']['x']
df['y']=np.argmax(data_RoI['train']['y'], axis=1)
value_counts = df['y'].value_counts()
print(value_counts)