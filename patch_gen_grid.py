import pandas as pd
import numpy as np
from openslide import OpenSlide
from multiprocessing import Pool, Value, Lock
import os
from skimage.color import rgb2hsv
from skimage.filters import threshold_otsu
from skimage.io import imsave, imread
from skimage.exposure.exposure import is_low_contrast
from skimage.transform import resize
from scipy.ndimage import binary_dilation, binary_erosion
import argparse
import logging
import pickle

from itertools import tee
from PIL import Image

def get_mask_image(img_RGB, RGB_min=50):
    img_HSV = rgb2hsv(img_RGB)

    background_R = img_RGB[:, :, 0] > threshold_otsu(img_RGB[:, :, 0])
    background_G = img_RGB[:, :, 1] > threshold_otsu(img_RGB[:, :, 1])
    background_B = img_RGB[:, :, 2] > threshold_otsu(img_RGB[:, :, 2])
    tissue_RGB = np.logical_not(background_R & background_G & background_B)
    tissue_S = img_HSV[:, :, 1] > threshold_otsu(img_HSV[:, :, 1])
    min_R = img_RGB[:, :, 0] > RGB_min
    min_G = img_RGB[:, :, 1] > RGB_min
    min_B = img_RGB[:, :, 2] > RGB_min

    mask = tissue_S & tissue_RGB & min_R & min_G & min_B
    return mask

def get_mask(slide, level='max', RGB_min=50):
    #read svs image at a certain level  and compute the otsu mask
    if level == 'max':
        level = len(slide.level_dimensions) - 1
    # note the shape of img_RGB is the transpose of slide.level_dimensions
    img_RGB = np.transpose(np.array(slide.read_region((0, 0),level,slide.level_dimensions[level]).convert('RGB')),
                           axes=[1, 0, 2])

    tissue_mask = get_mask_image(img_RGB, RGB_min)
    return tissue_mask, level


def extract_patches(slide_path, mask_path, patch_size, patches_output_dir, slide_id, max_patches_per_slide=2000):
    patch_folder = os.path.join(patches_output_dir, slide_id)
    
    if not os.path.isdir(patch_folder):
        os.makedirs(patch_folder)
    else:
        # Check if the directory is empty
        if not os.listdir(patch_folder):
            # The directory exists, but it's empty. Continue with the code.
            pass
        else:
            # The directory exists and contains files. Return from the function.
            return
        
    slide = OpenSlide(slide_path)


    patch_folder_mask = os.path.join(mask_path, slide_id)
    if not os.path.isdir(patch_folder_mask):
        os.makedirs(patch_folder_mask)
        mask, mask_level = get_mask(slide)
        mask = binary_dilation(mask, iterations=3)
        mask = binary_erosion(mask, iterations=3)
        np.save(os.path.join(patch_folder_mask, "mask.npy"), mask) 
    else:
        mask = np.load(os.path.join(mask_path, slide_id, 'mask.npy'))
        
    mask_level = len(slide.level_dimensions) - 1
    

    PATCH_LEVEL = 0
    BACKGROUND_THRESHOLD = .2

    try:
        #with open(os.path.join(patch_folder, 'loc.txt'), 'w') as loc:
        #loc.write("slide_id {0}\n".format(slide_id))
        #loc.write("id x y patch_level patch_size_read patch_size_output\n")

        ratio_x = slide.level_dimensions[PATCH_LEVEL][0] / slide.level_dimensions[mask_level][0]
        ratio_y = slide.level_dimensions[PATCH_LEVEL][1] / slide.level_dimensions[mask_level][1]

        xmax, ymax = slide.level_dimensions[PATCH_LEVEL]
        patch_size_resized = patch_size
        i = 0

        indices = [(x, y) for x in range(0, xmax, patch_size_resized[0]) for y in
                       range(0, ymax, patch_size_resized[0])]
        np.random.seed(5)
        np.random.shuffle(indices)

        for x, y in indices:
            # check if in background mask
            x_mask = int(x / ratio_x)
            y_mask = int(y / ratio_y)
            if mask[x_mask, y_mask] == 1:
                patch = slide.read_region((x, y), PATCH_LEVEL, patch_size_resized).convert('RGB')
                try:
                    mask_patch = get_mask_image(np.array(patch))
                    mask_patch = binary_dilation(mask_patch, iterations=3)
                except Exception as e:
                    print("error with slide id {} patch {}".format(slide_id, i))
                    print(e)
                if (mask_patch.sum() > BACKGROUND_THRESHOLD * mask_patch.size) and not (is_low_contrast(patch)):
                    #loc.write("{0} {1} {2} {3} {4} {5}\n".format(i, x, y, PATCH_LEVEL, patch_size_resized[0],
                    #                                                 patch_size_resized[1]))
                    imsave(os.path.join(patch_folder, "{0}_patch_{1}.jpeg".format(slide_id, i)), np.array(patch))
                    i += 1
                    
            if i >= max_patches_per_slide:
                break

        if i == 0:
            print("no patch extracted for slide {}".format(slide_id))



    except Exception as e:
        print("error with slide id {} patch {}".format(slide_id, i))
        print(e)

def get_slide_id(slide_name):
    return slide_name.split('.')[0]+'.'+slide_name.split('.')[1]


def process(opts):
    # global lock
    slide_list, patch_size, patches_output_dir_list, mask_path_list, slide_id_list, max_patches_per_slide = opts
    for slide_path, patches_output_dir, mask_path, slide_id in zip(slide_list, patches_output_dir_list, mask_path_list, slide_id_list):
        extract_patches(slide_path, mask_path, patch_size,
                    patches_output_dir, slide_id, max_patches_per_slide)


parser = argparse.ArgumentParser(description='Generate patches from a given folder of images')
parser.add_argument('--patch_size', default=768, type=int, help='patch size, '
                                                                'default 768')
parser.add_argument('--max_patches_per_slide', default=2000, type=int)
parser.add_argument('--num_process', default=10, type=int,
                    help='number of mutli-process, default 10')



if __name__ == '__main__':
    # count = Value('i', 0)
    # lock = Lock()

    args = parser.parse_args()
    ##DEBUG
    import pandas as pd
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
    #print("DEBUGING SMALL SLIDE LIST")
    #slide_list = ['GTEX-14A5I-0925.svs','GTEX-14A6H-0525.svs'
    #          ]
    def concatenar(row,texto='',wsi=False):
        if wsi:
            return 'BRACS_WSI/train/Group_'+ row['group'] + '/Type_' + row['WSI label']+'/'+ row['WSI Filename']+'.svs' 
        else:
            return 'BRACS_WSI'+texto+'/Group_'+ row['group'] + '/Type_' + row['WSI label']+'/'

    # Aplicar la función a las filas del DataFrame y crear una nueva columna 'Nombre completo'
    df_train['path'] = df_train.apply(lambda row: concatenar(row, wsi=True), axis=1)

    df_train['path_patch'] = df_train.apply(lambda row: concatenar(row, '_patches'), axis=1)
    df_train['mask_patch'] = df_train.apply(lambda row: concatenar(row, '_masks'), axis=1)
    slide_list=list(df_train['path'])
    slide_id=list(df_train['WSI Filename'])
    patch_path=list(df_train['path_patch'])
    mask_path=list(df_train['mask_patch'])
    opts = [
        slide_list, (args.patch_size, args.patch_size), patch_path, mask_path,
         slide_id, args.max_patches_per_slide]
    #pool = Pool(processes=args.num_process)
    #pool.map(process, opts)
    process(opts)
    '''
        from tqdm import tqdm
    for opt in tqdm(opts):
        process(opt)
    '''
