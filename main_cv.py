import os, sys

from model import *
from load_data import *
from train_utils import *

logfile = 'temp_'+get_date_str()+'.txt'
log_folder = os.getcwd()+'/logs_all'
os.system('mkdir -p '+log_folder)
rr_file = os.path.join(log_folder, logfile)

rr = Record_Results(rr_file)

modelc = '\nMODEL\n'+open('model.py').read()+'\n'
rr.fprint(modelc)

mainc = '\nMODEL\n' +open('main_cv.py').read()+'\n'
rr.fprint(mainc)


#Models without use conv

def default_config():
    config = {'batch_size':2, 'learning_rate':0.0001, 'momentum':0.5, 'reg_W':0., 'patience':10, 'num_epochs':100, 'NUM_LABELS':8, 'use_conv':True, 'logger':rr}
    config['total_data_points'] = 177
    config['conv_dict'] = {'static': True, 'MP': True, 'BN': True, 'SM': True, 'filter_size': 50, 'stride': 1,'pool_size': 1, 'sampling':False}
    #config['data'] = load_data()
    return config

logger = rr
use_convs = [False,True]
cv_ratios = [[9,1],[8,2],[2,1],[1,1]][:1]
models = ['SLAC']#,'CONCAT','MIX'][:2]
inp_types = ['TIF','TIF-MF', 'TIF-MF-CS', 'TIF-MF-CS-CRBG']
test_types = ['TIF', 'TIF-MF', 'TIF-MF-CS', 'TIF-MF-CS-CRBG']
comps = [True]

config = default_config()
config['test_types'] = test_types

for cv_ratio in cv_ratios:
    logger.fprint('Using a cross validation ratio of: ', cv_ratio)
    config['cv_ratio'] = cv_ratio
    for comp in comps:
        if comp:
            logger.fprint('Using composition')
        else:
            logger.fprint('Not using composition')
        config['comp'] = comp
        for use_conv in use_convs:
            if use_conv:    logger.fprint('Using conv for detecting peaks')
            else:   logger.fprint('Not usign conv for detecting peaks')
            config['use_conv'] = use_conv
            for model in models:
                logger.fprint('Current model is:', model)
                config['model'] = model
                if 'data' in config: del config['data']
                if 'train_data' in config: del config['train_data']
                if 'test_data' in config: del config['test_data']

                for inp_type in inp_types:
                    logger.fprint('Using input source: ', inp_type)
                    config['input_type'] = inp_type
                    if model=='MIX':
                        logger.fprint('Training with data aug and with mp opt')
                        config['mix_conf'] = {'opt': True, 'data_aug': True}
                        train_model_cv(config)
                        break
                        logger.fprint('Training without data aug and mp opt')
                        config['mix_conf'] = {'opt': False, 'data_aug': False}
                        train_model_cv(config)
                        logger.fprint('Training with data aug and without mp opt')
                        config['mix_conf'] = {'opt': False, 'data_aug': True}
                        train_model_cv(config)
                        logger.fprint('Training without data aug and with mp opt')
                        config['mix_conf'] = {'opt': True, 'data_aug': False}
                        train_model_cv(config)

                    else:
                        train_model_cv(config)


sys.exit(0)

# Try by making the conv static and dynamic

logger = rr
use_convs = [False, True]
cv_ratios = [[9,1],[8,2],[2,1],[1,1]]
models = ['CONCAT', 'MIX']
inp_types = ['TIF-MF-CS']
test_types = ['TIF','TIF-MF-CS']
comps = [False]

config = default_config()
config['test_types'] = test_types

for cv_ratio in cv_ratios:
    logger.fprint('Using a cross validation ratio of: ', cv_ratio)
    config['cv_ratio'] = cv_ratio
    for model in models:
        logger.fprint('Current model is:', model)
        config['model'] = model
        if 'data' in config: del config['data']
        if 'train_data' in config: del config['train_data']
        if 'test_data' in config: del config['test_data']
        for use_conv in use_convs:
            if use_conv:    logger.fprint('Using conv for detecting peaks')
            else:   logger.fprint('Not usign conv for detecting peaks')
            config['use_conv'] = use_conv
            for comp in comps:
                if comp: logger.fprint('Using composition')
                else: logger.fprint('Not using composition')
                config['comp'] = comp
                for inp_type in inp_types:
                    logger.fprint('Using input source: ', inp_type)
                    config['input_type'] = inp_type
                    if model=='MIX':
                        logger.fprint('Training with data aug and with mp opt')
                        config['mix_conf'] = {'opt': True, 'data_aug': True}
                        train_model_cv(config)
                        break
                        logger.fprint('Training without data aug and mp opt')
                        config['mix_conf'] = {'opt': False, 'data_aug': False}
                        train_model_cv(config)
                        logger.fprint('Training with data aug and without mp opt')
                        config['mix_conf'] = {'opt': False, 'data_aug': True}
                        train_model_cv(config)
                        logger.fprint('Training without data aug and with mp opt')
                        config['mix_conf'] = {'opt': True, 'data_aug': False}
                        train_model_cv(config)

                    else:
                        train_model_cv(config)
