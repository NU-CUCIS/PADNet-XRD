import os, sys, scipy.stats as stats
from load_data import *
from train_utils import *
import numpy as np
from sklearn.preprocessing import normalize

logfile = 'snr_'+get_date_str()+'.txt'
log_folder = os.getcwd()+'/logs_all'
os.system('mkdir -p '+log_folder)
rr_file = os.path.join(log_folder, logfile)

rr = Record_Results(rr_file)

#https://stackoverflow.com/questions/51413068/calculate-signal-to-noise-ratio-in-python-scipy-version-1-1

def mask(proc_img, radius):
    center = 1024
    X = np.arange(2048)
    Y = np.arange(2048)
    X, Y = np.meshgrid(X, Y)
    mask = (X-center)**2 + (Y-center)**2 > radius**2
    for p_img in range(proc_img.shape[0]):
        proc_img[p_img,...][mask] = 1
    return proc_img

mainc = '\nMODEL\n' +open('measure_snr.py').read()+'\n'
rr.fprint(mainc)


old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)


logger = rr
config = {}
config['logger'] = logger
config['model'] = 'MIX'

data_orig = None
inp_types = ['TIF']
for d_t in inp_types:
    logger.fprint(d_t)
    data_orig = load_data(config, [d_t], shuffle=False)

def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.mean(np.where(sd == 0, 0, m/sd))

logger.fprint('SNR for SLAC: ', signaltonoise(data_orig['SLAC-TIF']))
logger.fprint('SNR for Bruker: ', signaltonoise(data_orig['Bruker1-TIF']),signaltonoise(data_orig['Bruker2-TIF']))


test_types = ['TIF-MF', 'TIF-MF-CS', 'TIF-MF-CS-CRBG']
for t_t in test_types:
    print('\n\n'+t_t)
    data = None
    data = load_data(config, [t_t])
    data_source = ['SLAC', 'Bruker1', 'Bruker2']
    print(data.keys())
    for d_s in data_source:
        test_key = d_s+'-'+t_t
        out = data[test_key]
        orig = data_orig[d_s+'-TIF']
        bg = out-orig
        #logger.fprint('bg: ', bg.shape)
        bg = mask(bg, 1024-210)
        out = mask(out, 1024-210)
        bg = np.reshape(bg,(177,-1))
        out = np.reshape(out,(177,-1))
        logger.fprint('Data source: %s data_type: %s SNR: %.4f'%(d_s, t_t, signaltonoise(out)))


        # #out[out==0]=1
        # normalize(bg, axis=1, copy=False)
        # normalize(out, axis=1, copy=False)
        # var_bg = np.var(bg, axis=0)
        # var_out = np.var(out, axis=0)
        # var_bg[var_bg==0] = 0
        # snr = np.mean(np.divide(var_out, var_bg))
        # logger.fprint('Data source: %s data_type: %s SNR: %.4f'%(d_s, t_t, snr))






