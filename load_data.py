import os
from scipy import misc
from PIL import Image
import numpy as np
from sklearn import preprocessing
import math

dir_path = '/raid/dkj755/XRD'

def load_slac_data(dest_path):
    print 'loading from: ', dest_path
    slac_data = []
    slac_path = os.path.join(dir_path, dest_path)
    for i in range(1,178):
        fname = str(i)+'.tif'
        img = misc.imread(os.path.join(slac_path, fname), flatten=0)
        slac_data.append(img)
    slac_data = np.asarray(slac_data).astype(np.float64)#.swapaxes(3,1).swapaxes(2,3).astype(np.float64)
    slac_shape = slac_data.shape
    slac_data = np.reshape(slac_data, (slac_shape[0],slac_shape[1], slac_shape[2],1))
    print slac_data.shape
    return slac_data

def load_bruker_data(dest_path):
    print 'loading from: ', dest_path
    bruker_data = [[],[]]
    bruker_path = os.path.join(dir_path, dest_path)
    for i in range(1,178):
        fname = str(i)+'_1.tif'
        img = misc.imread(os.path.join(bruker_path, fname), flatten=0)
        bruker_data[0].append(img)
        fname = str(i) + '_2.tif'
        img = misc.imread(os.path.join(bruker_path, fname), flatten=0)
        bruker_data[1].append(img)
    bruker_data = np.asarray(bruker_data).astype(np.float64)#.swapaxes(3,1).swapaxes(2,3).astype(np.float64)
    bruker_shape = bruker_data.shape
    bruker_data = np.reshape(bruker_data, (bruker_shape[0],bruker_shape[1], bruker_shape[2], bruker_shape[3],1))
    print bruker_data.shape
    return bruker_data[0,...], bruker_data[1,...]

def load_comp_data():
    comp_data = []
    comp_file = os.path.join(dir_path, 'wafer_data', 'Composition.txt')
    comp_fd = open(comp_file).read().strip().split('\n')[1:]
    #print comp_fd[0]
    for cd in comp_fd:
        cd = [float(x) for x in  cd.split('\t')]
        comp_data.append(cd)
    return np.asarray(comp_data)


def load_labels():
    comp_data = []
    comp_file = os.path.join(dir_path, 'wafer_data', 'wafer_labels.txt')
    comp_fd = open(comp_file).read().strip().split('\n')
    #print comp_fd[0]
    for cd in comp_fd:
        cd = int(cd)-1
        comp_data.append(cd)
    comp_data = [x-1 if x > 0 else x for x in comp_data]
    return np.asarray(comp_data)


slac_dirs = ['SLAC_images-TIF', 'SLAC_images-TIF-MF', 'SLAC_images-TIF-MF-CS', 'SLAC_images-TIF-MF-CS-CRBG']
bruker_dirs = ['Bruker_images-TIF', 'Bruker_images-TIF-MF', 'Bruker_images-TIF-MF-CS', 'Bruker_images-TIF-MF-CS-CRBG']

def load_data(config, data_types, shuffle=False):
    logger = config['logger']
    logger.fprint('Loading data', data_types)
    model = config['model']
    config['data'] = None
    #data = None
    #if 'data' in config: data = config['data']
    #if data is None: data = {}
    data = {}
    for s_dir in slac_dirs:
        if not any([x for x in data_types if s_dir.endswith(x)]): continue
        if not model in ['SLAC', 'MIX', 'CONCAT']: continue
        data_key = s_dir.replace('_images','')
        #if not data_key in data or data[data_key] is None:
        dest_path = os.path.join(dir_path, s_dir)
        data[data_key] = load_slac_data(dest_path)
    for s_dir in bruker_dirs:
        if not any([x for x in data_types if s_dir.endswith(x)]): continue
        if not model in ['Bruker', 'MIX', 'CONCAT']: continue
        data_key = s_dir.replace('_images', '1')
        #if not data_key in data or data[data_key] is None:
        dest_path = os.path.join(dir_path, s_dir)
        data[s_dir.replace('_images','1')], data[s_dir.replace('_images','2')] = load_bruker_data(      dest_path)

    #if config['comp']:
    #if not 'comp' in data:
    data['comp'] = load_comp_data()
    #if not 'labels' in data:
    data['labels'] = load_labels()
    config['data'] = data

    logger.fprint(data.keys())
    if shuffle:
        randomize = np.arange(data['SLAC-TIF'].shape[0])
        np.random.shuffle(randomize)
        for dk in data.keys():
            data[dk] = data[dk][randomize,...]
    return data

#load_data()
