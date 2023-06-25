import os
import pickle
import argparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Crear el objeto ArgumentParser y definir los argumentos
parser = argparse.ArgumentParser(description='Configuración para la visualización de resultados')

parser.add_argument('--results_folder_name', type=str, default='resultados_v2',
                    help='Nombre de la carpeta para los resultados')

# Parsear los argumentos
args = parser.parse_args()
results_folder_name = args.results_folder_name

path_dir='./'
save_path = path_dir+'results/'+results_folder_name+'/'
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


Prob=1

for i in ['val', 'train']:
  data = pd.DataFrame()
  data['Case_Ids'] = results['Case_Ids']
  if i=="train":
    data['Preds'] = results['train_preds']
    data['Real'] =  results['train_labels']
    probs=results['train_probs']

  else:
    data['Preds'] = results['val_preds']
    data['Real'] =  results['val_labels']
    probs=results['val_probs']

  ids=[]
  for j in data['Case_Ids']:
    aux='_'.join(j.split('_')[0:-1])
    aux=aux+'_'
    ids.append(aux)
  ids=pd.unique(ids)
  final=pd.DataFrame(columns=['Case_id','preds','real'])
  y_true=[]
  y_pred=[]

  if Prob:
    for k in ids:
        p = data[data['Case_Ids'].str.contains(k)]
        m_train=probs[p.index]
        pred=np.argmax(m_train.sum(axis=0))
        real=p['Real'].value_counts().idxmax()
        labels = str(np.where(real == 0, 'AT', np.where(real == 1, 'BT', 'MT')))
        preds= str(np.where(pred == 0, 'AT', np.where(pred == 1, 'BT', 'MT')).astype(str))
        final=final.append({'Case_id':k,'preds':preds,'real':labels}, ignore_index=True)
    final.to_excel(save_path+i+'_results'+'.xlsx')
    accuracy = accuracy_score( np.array(final['real']), np.array(final['preds']))
    f1 = f1_score( np.array(final['real']), np.array(final['preds']), average='weighted')
    cm=confusion_matrix(np.array(final['real']), np.array(final['preds']))
    text_acc=i+' accuracy:'+ str(accuracy)
    text_f1=i+' f1 score:'+ str(f1)
    print(text_acc) 
    print(text_f1) 
    print('Matriz de confusión: ')
    print(cm)
    name='matriz_confusion'+'_'+i+'.png'
    disp=ConfusionMatrixDisplay(cm, display_labels=['AT', 'BT', 'MT'])
    disp.plot()
    plt.savefig(name)
    plt.show()
    # Guardar la visualización como un archivo PNG
    plt.savefig('matriz_confusion.png')

  else:
    for k in ids:
      p=data[data['Case_Ids'].str.contains(k)]
      pred=p['Preds'].value_counts().idxmax()
      real=p['Real'].value_counts().idxmax()
      labels = str(np.where(real == 0, 'AT', np.where(real == 1, 'BT', 'MT')))
      preds= str(np.where(pred == 0, 'AT', np.where(pred == 1, 'BT', 'MT')).astype(str))
      final=final.append({'Case_id':k,'preds':preds,'real':labels}, ignore_index=True)
    final.to_excel(save_path+i+'_results'+'.xlsx')
    accuracy = accuracy_score( np.array(final['real']), np.array(final['preds']))
    f1 = f1_score( np.array(final['real']), np.array(final['preds']), average='weighted')
    cm=confusion_matrix(np.array(final['real']), np.array(final['preds']))
    text_acc=i+' accuracy:'+ str(accuracy)
    text_f1=i+' f1 score:'+ str(f1)
    print(text_acc) 
    print(text_f1) 
    print('Matriz de confusión: ')
    print(cm) 
    name='matriz_confusion'+'_'+i+'.png'
    disp=ConfusionMatrixDisplay(cm, display_labels=['AT', 'BT', 'MT'])
    disp.plot()
    plt.savefig(name)
    plt.show()