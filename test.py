import os
import pickle
import argparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pytorch_datasets import Dataset, Dataset_full
from train_pred import predict_WSI
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms as transforms

# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para la visualización de resultados')

parser.add_argument('--results_folder_name', type=str, default='resultados_v2',
                    help='Nombre de la carpeta para los resultados')
parser.add_argument('--data_RoI', type=str, default='data_RoI_256.pkl',
                    help='pkl de datasets')
parser.add_argument('--Prob', type=int, default=1,
                    help='Indicador booleano para habilitar o deshabilitar los pesos segun el tamaño de la clase')
parser.add_argument('--normalization', type=str, default='macenko',
                    help='pkl de datasets')
parser.add_argument('--full', type=int, default=0,
                    help='Indicador booleano para habilitar o deshabilitar tratar con full imagenes')
parser.add_argument('--im_size', type=int, default=512,
                    help='Tamaño de la imagen')
parser.add_argument('--n_clases', type=int, default=3,
                    help='Número de clases')

# Parsear los argumentos
args = parser.parse_args()
results_folder_name = args.results_folder_name
data_RoI_pkl=args.data_RoI
Prob=bool(args.Prob)
norm=args.normalization
path_dir='./'
save_path = path_dir+'results/'+results_folder_name+'/'
directorio_actual = os.getcwd()
full=bool(args.full)
n = args.im_size
n_clases=args.n_clases

val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
""" Crear readers """
dataReaders = {}

with open(data_RoI_pkl, 'rb') as fp:
    data_RoI = pickle.load(fp)
dataReaders['CNN'] = data_RoI


import warnings
warnings.filterwarnings("ignore")
#crear test dataloader

if full: 
    dataset_test = Dataset_full(dataReaders['CNN']['test']['x'],
                        dataReaders['CNN']['test']['y'], val_transform, normalization=norm, resize=(n,n))
else:
    dataset_test = Dataset(dataReaders['CNN']['test']['x'],
                        dataReaders['CNN']['test']['y'], val_transform, normalization=norm)
    
dataloader_test = DataLoader(dataset_test, batch_size=1,
                                shuffle=False, num_workers=1,
                                pin_memory=True)

dataloaders = { 'test': dataloader_test}
#obtener los tamaños de los datasets
dataset_sizes = {x: len(dataReaders['CNN'][x]['x']) for x in [ 'train', 'val','test']}

os.chdir(save_path) 
# Obtener la lista de archivos en la carpeta actual
archivos = os.listdir()

# Buscar el archivo que cumpla con el criterio
archivo_deseado = None
for archivo in archivos:
    if archivo.startswith("results_Epoch_") and archivo.endswith(".pkl"):
        archivo_deseado = archivo
        break

# Verificar si se encontró el archivo
if archivo_deseado is not None:
    # Abrir el archivo
    with open(archivo_deseado, 'rb') as file:
        results = pickle.load(file)

    # Realizar las operaciones necesarias con los datos del archivo
    # ...
else:
    print("No se encontró ningún archivo que cumpla con el criterio.")

model = results['model']

model.eval()

os.chdir(directorio_actual)
# predecir en el conjunto de pruebas 
test_results = predict_WSI(model, dataloader_test, dataset_sizes['test'])
print('Test acc: {:.4f}\n'.format(test_results['acc']))

# escribir los resultados del dict en un archivo
a_file = open(save_path+'test_results.pkl', "wb")
pickle.dump(test_results, a_file)
a_file.close()

test_labels = np.argmax(dataReaders['CNN']['test']['y'], axis=1)
case_ids_test = dataReaders['CNN']['test']['x']

data = pd.DataFrame()
data['Case_Ids'] = case_ids_test
data['Preds'] = test_results['preds']
data['Real'] = test_labels

data.to_excel(save_path+'test.xlsx')

#Prediccion de las imagenes completas

data = pd.DataFrame()
data['Case_Ids'] = dataReaders['CNN']['test']['x']
data['Preds'] = test_results['preds']
data['Real'] =  test_results['labels']
probs=test_results['probs']
i='test'

if n_clases==3:
            clases=['AT', 'BT', 'MT']
else:
            clases=['N', 'PB', 'UDH', 'FEA', 'ADH', 'DCIS', 'IC']

if full:
    y_real=np.array(data['Real'])
    y_pred=np.array(data['Preds'])
else:
    ids=[]
    for j in data['Case_Ids']:
        aux='_'.join(j.split('_')[0:-1])
        aux=aux+'_'
        ids.append(aux)
    ids=pd.unique(ids)
    final=pd.DataFrame(columns=['Case_id','preds','real'])

    for k in ids:
        p = data[data['Case_Ids'].str.contains(k)]
        if Prob:
            m_train=probs[p.index]
            pred=np.argmax(m_train.sum(axis=0))
        else: 
            pred=p['Preds'].value_counts().idxmax()
        real=p['Real'].value_counts().idxmax()

        label_mapping = dict(zip(range(n_clases), clases))
        # Mapear los valores reales a etiquetas
        labels = str([label_mapping[value] for value in real])
        preds= str([label_mapping[value] for value in pred])
        final=final.append({'Case_id':k,'preds':preds,'real':labels}, ignore_index=True)
    final.to_excel(save_path+i+'_results'+'.xlsx')
    y_real= np.array(final['real'])
    y_pred= np.array(final['preds'])

accuracy = accuracy_score(y_real, y_pred)

f1_W = f1_score(y_real, y_pred, average='weighted')
f1_micro = f1_score(y_real, y_pred, average='micro')
f1_macro = f1_score(y_real, y_pred, average='macro')

cm=confusion_matrix(y_real, y_pred)
text_acc=i+' accuracy:'+ str(accuracy)
text_f1w=i+' f1 score weighted:'+ str(f1_W)
text_f1mi=i+' f1 score micro:'+ str(f1_micro)
text_f1ma=i+' f1 score macro:'+ str(f1_macro)
print(text_acc) 
print(text_f1w) 
print(text_f1mi) 
print(text_f1ma) 
print('Matriz de confusión: ')
print(cm)
name='matriz_confusion_test_nclases'+str(n_clases)+'.png'
disp=ConfusionMatrixDisplay(cm, display_labels=clases)
disp.plot()
plt.show()
# Guardar la visualización como un archivo PNG
os.chdir(save_path) 
plt.savefig(name)
