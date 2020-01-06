import tensorflow as tf, numpy as np
from load_data import *
import time
from train_utils import *

SEED = 1234567
tf.set_random_seed(SEED)
np.random.seed(SEED)



def load_config(config):
    global batch_size, learning_rate, momentum, reg_W, patience, num_epochs, NUM_LABELS, use_conv, logger
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    reg_W = config['reg_W']
    patience = config['patience']
    num_epochs = config['num_epochs']
    NUM_LABELS = config['NUM_LABELS']
    use_conv = config['use_conv']
    logger = config['logger']

def model_slim(data, architecture, train=True,scope = 'graph_'):
    i=0
    if train:
        reuse = None
    else:
        reuse = True
    nets = {}
    nets[0] = data
    with tf.variable_scope(scope, reuse = tf.AUTO_REUSE):
        for arch in architecture:
            #logger.fprint(i, nets[i].get_shape())
            i +=1
            layer_type = arch['layer_type']
            if layer_type == 'conv':
                logger.fprint ('adding cnn layer..', i)
                num_filters = arch['num_filters']
                filter_size = arch['filter_size']
                border_mode = 'same'
                activation = tf.nn.relu
                if arch.has_key('border_mode'):
                    border_mode = arch['border_mode']
                padding=border_mode
                if arch.has_key('padding'):
                    padding = arch['padding']
                if arch.has_key('activation'):
                    if arch['activation'] == 'sigmoid':
                        activation = tf.nn.sigmoid
                stride = 1
                if arch.has_key('stride'):
                    stride = arch['stride']
                weights_initializer = tf.truncated_normal_initializer(stddev=0.05)
                nets[i] = tf.layers.conv2d(nets[i-1], filters=num_filters,kernel_size=[filter_size, filter_size], kernel_initializer=weights_initializer, padding=padding, name=scope+'conv'+str(i), strides=(stride, stride), kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_W), reuse=reuse, activation=activation)
                if 'BN' in arch and arch['BN']:
                    nets[i] = tf.layers.batch_normalization(nets[i], axis=1)


            elif layer_type == 'fully_connected':
                num_outputs = arch['num_outputs']
                activation = tf.nn.relu
                if arch['activation'] == 'sigmoid':
                    activation = tf.nn.sigmoid
                elif arch['activation'] =='linear':
                    activation = None
                #logger.fprint ('adding fully connected layer...', i, ' with ', num_outputs)

                nets[i] = tf.layers.dense(nets[i-1], units=num_outputs, name=scope+'fc'+str(i),activation=activation, reuse=reuse)
                if 'BN' in arch and arch['BN']:
                    nets[i] = tf.layers.batch_normalization(nets[i], axis=1)

            elif layer_type == 'AvgPool2D':
                #logger.fprint ('adding avg pooling...', i, ' with ', arch['pool_size'])

                nets[i] = tf.layers.average_pooling2d(nets[i-1], [arch['pool_size'], arch['pool_size']])

            elif layer_type == 'maxpool2D':
                #logger.fprint ('adding max pooling...', i, ' with ', arch['pool_size'])

                nets[i] = tf.layers.max_pooling2d(nets[i - 1], pool_size=[arch['pool_size'], arch['pool_size']], strides = [arch['pool_size'], arch['pool_size']])

            elif layer_type == 'flatten':
                #logger.fprint ('adding flattening...', i)

                nets[i] = tf.layers.flatten(nets[i-1], name=scope+'flatten'+str(i))
            elif layer_type == 'dropout':
                #logger.fprint('adding dropout with ', arch['value'])
                nets[i] = tf.nn.dropout(nets[i-1], arch['value'], seed=SEED)
        #logger.fprint('final shape:', nets[i].get_shape())
    return nets[i]


class Model:    pass

def conv_input(net, train=True, scope='graph_', conv_dict=None):
    orig_net = net
    if not conv_dict:
        conv_dict = {'static':True, 'MP':True,'BN':True,'SM':True, 'filter_size':50, 'stride':1, 'pool_size':1}
    filter_size = conv_dict['filter_size']
    stride = conv_dict['stride']
    pool_size = conv_dict['pool_size']
    logger.fprint('defining conv filters for preprocessing')
    #diagonal filters
    diag_filt1 = np.asarray([[-1. if x < y else 1. for x in range(-filter_size//2, filter_size//2)] for y in range(-filter_size//2, filter_size//2)], dtype=np.float32)
    diag_filt2 = np.asarray(
        [[-1. if x + y > filter_size else 1. for x in range(-filter_size // 2, filter_size // 2 )] for y in
         range(-filter_size // 2, filter_size // 2 )], dtype=np.float32)
    vert_filt = np.asarray([[-1. if x < 0 else 1. for x in range(-filter_size//2, filter_size//2)] for y in range(-filter_size//2, filter_size//2)], dtype=np.float32)
    horz_filt = np.asarray([[-1. if y < 0 else 1. for x in range(-filter_size // 2, filter_size // 2)] for y in
                           range(-filter_size // 2, filter_size // 2)], dtype=np.float32)
    filters = [diag_filt1, diag_filt2, vert_filt, horz_filt]

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if conv_dict['static']:
            filters = [tf.constant(x.reshape((filter_size, filter_size, 1, 1))) for x in filters]
        else:
            filters = [tf.Variable(x.reshape((filter_size, filter_size, 1, 1))) for x in filters]
        filters = tf.concat(filters, axis=3)
        logger.fprint(net.get_shape())
        net = tf.nn.conv2d(net, filters,strides=[1,stride,stride,1], padding='SAME')
        net = tf.abs(net)
        logger.fprint(net.get_shape())
        if conv_dict['MP']:
            net = tf.nn.max_pool(net, ksize=[1,pool_size, pool_size, 1], strides=[1,1,1,1], padding='SAME')
        logger.fprint(net.get_shape())
        if conv_dict['BN']:
            net = tf.layers.batch_normalization(net, axis=-1)
        if conv_dict['SM']:
            net = tf.nn.softmax(net)
        logger.fprint('orig_net shape:',orig_net.get_shape().as_list())
        # if conv_dict['sampling']:
        #     logger.fprint('sampling..')
        #     sampled_nets = []
        #     for l in range(net.get_shape().as_list()[-1]):
        #         print(net[:,:,:,l].get_shape())
        #         sampled_nets.append(tf.multiply(orig_net, net[:,:,:,l:l+1]))
        #     sampled_nets = tf.concat(sampled_nets, axis=3)
        #     logger.fprint(sampled_nets.get_shape())
        #     net = sampled_nets

    return net


def slac_conv_graph(net,train=True, scope='graph_', input_channels = 1):
    arch = [{'layer_type':'conv', 'num_filters':64, 'input_channels':input_channels, 'filter_size':5, 'border_mode':'same', 'init':'glorot_uniform', 'stride':4,'activation':'relu', 'reg_W':reg_W, 'BN':True},
            {'layer_type':'maxpool2D', 'pool_size':2},
            {'layer_type': 'conv', 'num_filters': 96, 'input_channels': 64, 'filter_size': 5, 'border_mode': 'same',
             'init': 'glorot_uniform', 'stride': 2, 'activation': 'relu', 'reg_W': reg_W, 'BN':True},
            {'layer_type': 'maxpool2D', 'pool_size': 2},

            {'layer_type': 'conv', 'num_filters': 32, 'input_channels': 96, 'filter_size': 5, 'border_mode': 'same',
             'init': 'glorot_uniform', 'stride': 2, 'activation': 'relu', 'reg_W': reg_W, 'BN':True},
            {'layer_type': 'maxpool2D', 'pool_size': 2},
            {'layer_type':'flatten'},
            {'layer_type': 'fully_connected', 'num_outputs': 512,
             'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform', 'BN':True}
            ]

    return model_slim(net, arch, train=train, scope=scope)


def bruker_conv_graph(net,train=True, scope='graph_', input_channels=1):
    arch = [{'layer_type': 'conv', 'num_filters': 32, 'input_channels': input_channels, 'filter_size': 5, 'border_mode': 'same',
             'init': 'glorot_uniform', 'stride': 4, 'activation': 'relu', 'reg_W': reg_W, 'BN':True},
            {'layer_type': 'maxpool2D', 'pool_size': 2},

            {'layer_type': 'conv', 'num_filters': 48, 'input_channels': 32, 'filter_size': 5, 'border_mode': 'same',
             'init': 'glorot_uniform', 'stride': 2, 'activation': 'relu', 'reg_W': reg_W, 'BN':True},{'layer_type':'maxpool2D', 'pool_size':2},


            {'layer_type': 'conv', 'num_filters': 16, 'input_channels': 48, 'filter_size': 5, 'border_mode': 'same',
             'init': 'glorot_uniform', 'stride': 2, 'activation': 'relu', 'reg_W': reg_W, 'BN':True},
            {'layer_type': 'maxpool2D', 'pool_size': 2},
            {'layer_type': 'flatten'},

            {'layer_type': 'fully_connected', 'num_outputs': 256,
             'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform', 'BN':True},
            ]

    return model_slim(net, arch, train=train, scope=scope)

def dense_layers(net, train=True, scope='graph_'):
    arch = [{'layer_type': 'fully_connected', 'num_outputs': 256,
             'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform', 'BN':True},
            #{'layer_type': 'fully_connected', 'num_outputs': 512,
            # 'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform', 'BN':True},
            {'layer_type': 'fully_connected', 'num_outputs': NUM_LABELS,
             'activation': 'linear', 'reg_W': reg_W, 'init': 'glorot_uniform'},

            ]
    return model_slim(net, arch, train=train, scope=scope)


def comp_graph(net, train=True, scope='graph_'):
    arch = [{'layer_type': 'fully_connected', 'num_outputs': 256, 'num_inputs': 3,
                     'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform', 'BN':True},
            {'layer_type': 'fully_connected', 'num_outputs': 256, 'num_inputs': 256,
             'activation': 'relu', 'reg_W': reg_W, 'init': 'glorot_uniform', 'BN':True}
            ]
    return model_slim(net, arch, train=train, scope=scope)




def create_comp_model(config, comp=True):
    logger.fprint('Creating a comp model')
    scope = 'comp_'
    model = Model()
    model.labels_node = tf.placeholder(tf.int64, shape=batch_size)
    model.inp_ph = {'labels':model.labels_node}
    model.eval_ph = {}
    if comp:
        logger.fprint('Building composition graph')
        model.comp_data_node = tf.placeholder(tf.float32, shape=[batch_size,3])
        model.eval_comp_data_node = tf.placeholder(tf.float32, shape=[batch_size,3])
        model.inp_ph['comp']=model.comp_data_node
        model.eval_ph['comp'] = model.eval_comp_data_node
        model.comp_net = comp_graph(model.comp_data_node, scope=scope+'_comp_')
        model.eval_comp_net = comp_graph(model.eval_comp_data_node, train=False, scope=scope+'_comp_')
        net = model.comp_net
        eval_net = model.eval_comp_net
    logger.fprint('Building classifier layers')
    model.logits = dense_layers(net,scope=scope)
    model.eval_logits = dense_layers(eval_net, train=False, scope=scope)
    logger.fprint('Defining loss and optimizer')
    model.loss = tf.losses.sparse_softmax_cross_entropy(model.labels_node, model.logits)
    model.loss=tf.reduce_mean(model.loss)
    model.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    model.eval_logits = tf.nn.softmax(model.eval_logits)
    return model

def create_slac_model(config, comp=True):
    logger.fprint('Creating a slac model')
    scope = 'slac_graph_'
    model = Model()
    model.slac_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048,2048,1])
    model.eval_slac_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048,1])
    model.labels_node = tf.placeholder(tf.int64, shape=batch_size)
    net = model.slac_data_node
    eval_net = model.eval_slac_data_node
    model.inp_ph = {'SLAC':model.slac_data_node, 'labels':model.labels_node}
    model.eval_ph = {'SLAC': model.eval_slac_data_node}
    input_channels=1
    if use_conv:
        logger.fprint('Using conv to detect peaks')
        input_channels=4
        net = conv_input(net, train=True, conv_dict=config['conv_dict'])
        eval_net = conv_input(eval_net, train=False, conv_dict = config['conv_dict'])
    logger.fprint('Building conv graph for slac image')
    model.slac_conv_net = slac_conv_graph(net,scope=scope+'_conv_', input_channels=input_channels)
    model.eval_slac_conv_net = slac_conv_graph(eval_net, train=False, scope=scope+'_conv_', input_channels=input_channels)
    net = tf.layers.flatten(model.slac_conv_net,name='slac_flatten')
    eval_net = tf.layers.flatten(model.eval_slac_conv_net, name='slac_flatten')
    if comp:
        logger.fprint('Building composition graph')
        model.comp_data_node = tf.placeholder(tf.float32, shape=[batch_size,3])
        model.eval_comp_data_node = tf.placeholder(tf.float32, shape=[batch_size,3])
        model.inp_ph['comp']=model.comp_data_node
        model.eval_ph['comp'] = model.eval_comp_data_node
        model.comp_net = comp_graph(model.comp_data_node, scope=scope+'_comp_')
        model.eval_comp_net = comp_graph(model.eval_comp_data_node, train=False, scope=scope+'_comp_')
        net = tf.concat([net, model.comp_net], axis=1)
        eval_net = tf.concat([eval_net, model.eval_comp_net], axis=1)
    logger.fprint('Building classifier layers')
    model.logits = dense_layers(net,scope=scope)
    model.eval_logits = dense_layers(eval_net, train=False, scope=scope)
    logger.fprint('Defining loss and optimizer')
    model.loss = tf.losses.sparse_softmax_cross_entropy(model.labels_node, model.logits)
    model.loss=tf.reduce_mean(model.loss)
    model.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    model.eval_logits = tf.nn.softmax(model.eval_logits)
    return model

def create_bruker_model(config,comp=True):
    logger.fprint('Creating a bruker model')
    scope = 'bruker_graph_'
    model = Model()
    input_channels=1
    model.bruker1_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1])
    model.eval_bruker1_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1])
    model.bruker2_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1])
    model.eval_bruker2_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1])
    model.labels_node = tf.placeholder(tf.int64, shape=batch_size)
    net1 = model.bruker1_data_node
    eval_net1 = model.eval_bruker1_data_node
    net2 = model.bruker2_data_node
    eval_net2 = model.eval_bruker2_data_node

    model.inp_ph = {'Bruker1': model.bruker1_data_node, 'Bruker2':model.bruker2_data_node, 'labels': model.labels_node}
    model.eval_ph = {'Bruker1': model.eval_bruker1_data_node, 'Bruker2':model.eval_bruker2_data_node}
    if use_conv:
        logger.fprint('Using conv to detect peaks')
        input_channels=4
        net1 = conv_input(net1, train=True, conv_dict=config['conv_dict'])
        eval_net1 = conv_input(eval_net1, train=False, conv_dict=config['conv_dict'])
        net2 = conv_input(net2, train=True, conv_dict=config['conv_dict'])
        eval_net2 = conv_input(eval_net2, train=False, conv_dict=config['conv_dict'])

    logger.fprint('Building conv graph for bruker image')
    model.bruker1_conv_net = bruker_conv_graph(net1, scope=scope + '_1_conv_', input_channels=input_channels)
    model.eval_bruker1_conv_net = bruker_conv_graph(eval_net1, train=False, scope=scope + '_1_conv_', input_channels=input_channels)
    model.bruker2_conv_net = bruker_conv_graph(net2, scope=scope + '_2_conv_', input_channels=input_channels)
    model.eval_bruker2_conv_net = bruker_conv_graph(eval_net2, train=False, scope=scope + '_2_conv_', input_channels=input_channels)
    net1 = tf.layers.flatten(model.bruker2_conv_net, name='bruker1_flatten')
    eval_net1 = tf.layers.flatten(model.eval_bruker1_conv_net, name='slac_flatten')
    net2 = tf.layers.flatten(model.bruker2_conv_net, name='bruker2_flatten')
    eval_net2 = tf.layers.flatten(model.eval_bruker2_conv_net, name='slac_flatten')
    net = tf.concat([net1, net2], axis=1)
    eval_net = tf.concat([eval_net1, eval_net2], axis=1)
    if comp:
        logger.fprint('Building composition graph')
        model.comp_data_node = tf.placeholder(tf.float32, shape=[batch_size, 3])
        model.eval_comp_data_node = tf.placeholder(tf.float32, shape=[batch_size, 3])
        model.inp_ph['comp'] = model.comp_data_node
        model.eval_ph['comp'] = model.eval_comp_data_node
        model.comp_net = comp_graph(model.comp_data_node, scope=scope + '_comp_')
        model.eval_comp_net = comp_graph(model.eval_comp_data_node, train=False, scope=scope + '_comp_')
        net = tf.concat([net, model.comp_net], axis=1)
        eval_net = tf.concat([eval_net, model.eval_comp_net], axis=1)
    logger.fprint('Building classifier layers')
    model.logits = dense_layers(net, scope=scope)
    model.eval_logits = dense_layers(eval_net, train=False, scope=scope)
    logger.fprint('Defining loss and optimizer')
    model.loss = tf.losses.sparse_softmax_cross_entropy(model.labels_node, model.logits)
    model.loss = tf.reduce_mean(model.loss)
    model.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    model.eval_logits = tf.nn.softmax(model.eval_logits)
    return model


def create_concat_model(config,comp=True):
    logger.fprint('Creating a CONCAT model')
    scope = 'slac_graph_'
    model = Model()
    model.slac_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='slac_ph')
    model.eval_slac_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='eval_slac_ph')
    net = model.slac_data_node
    eval_net = model.eval_slac_data_node
    model.bruker1_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='bruker1_ph')
    model.eval_bruker1_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='eval_bruker1_ph')
    model.bruker2_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='bruker2_ph')
    model.eval_bruker2_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='eval_bruker2_ph')
    model.labels_node = tf.placeholder(tf.int64, shape=batch_size, name='labels')
    net1 = model.bruker1_data_node
    eval_net1 = model.eval_bruker1_data_node
    net2 = model.bruker2_data_node
    eval_net2 = model.eval_bruker2_data_node
    input_channels=1

    model.inp_ph = {'Bruker1': model.bruker1_data_node, 'Bruker2': model.bruker2_data_node, 'labels': model.labels_node, 'SLAC':model.slac_data_node}
    model.eval_ph = {'Bruker1': model.eval_bruker1_data_node, 'Bruker2': model.eval_bruker2_data_node, 'SLAC':model.eval_slac_data_node}

    if use_conv:
        logger.fprint('Using conv to detect peaks')
        input_channels=4
        net = conv_input(net, train=True, conv_dict=config['conv_dict'])
        eval_net = conv_input(eval_net, train=False, conv_dict=config['conv_dict'])
        net1 = conv_input(net1, train=True, conv_dict=config['conv_dict'])
        eval_net1 = conv_input(eval_net1, train=False, conv_dict=config['conv_dict'])
        net2 = conv_input(net2, train=True, conv_dict=config['conv_dict'])
        eval_net2 = conv_input(eval_net2, train=False, conv_dict=config['conv_dict'])

    logger.fprint('Building conv graph for slac image')
    model.slac_conv_net = slac_conv_graph(net, scope=scope + '_conv_', input_channels=input_channels)
    model.eval_slac_conv_net = slac_conv_graph(eval_net, train=False, scope=scope + '_conv_', input_channels=input_channels)
    net = tf.layers.flatten(model.slac_conv_net, name='slac_flatten')
    eval_net = tf.layers.flatten(model.eval_slac_conv_net, name='slac_flatten')

    logger.fprint('Building conv graph for bruker image')
    model.bruker1_conv_net = bruker_conv_graph(net1, scope=scope + '_1_conv_', input_channels=input_channels)
    model.eval_bruker1_conv_net = bruker_conv_graph(eval_net1, train=False, scope=scope + '_1_conv_', input_channels=input_channels)
    model.bruker2_conv_net = bruker_conv_graph(net2, scope=scope + '_2_conv_', input_channels=input_channels)
    model.eval_bruker2_conv_net = bruker_conv_graph(eval_net2, train=False, scope=scope + '_2_conv_', input_channels=input_channels)
    net1 = tf.layers.flatten(model.bruker2_conv_net, name='bruker1_flatten')
    eval_net1 = tf.layers.flatten(model.eval_bruker1_conv_net, name='slac_flatten')
    net2 = tf.layers.flatten(model.bruker2_conv_net, name='bruker2_flatten')
    eval_net2 = tf.layers.flatten(model.eval_bruker2_conv_net, name='slac_flatten')
    net = tf.concat([net,net1, net2], axis=1)
    eval_net = tf.concat([eval_net,eval_net1, eval_net2], axis=1)

    if comp:
        logger.fprint('Building composition graph')
        model.comp_data_node = tf.placeholder(tf.float32, shape=[batch_size, 3])
        model.eval_comp_data_node = tf.placeholder(tf.float32, shape=[batch_size, 3])
        model.inp_ph['comp'] = model.comp_data_node
        model.eval_ph['comp'] = model.eval_comp_data_node
        model.comp_net = comp_graph(model.comp_data_node, scope=scope + '_comp_')
        model.eval_comp_net = comp_graph(model.eval_comp_data_node, train=False, scope=scope + '_comp_')
        net = tf.concat([net, model.comp_net], axis=1)
        eval_net = tf.concat([eval_net, model.eval_comp_net], axis=1)

    logger.fprint('Building classifier layers')
    model.logits = dense_layers(net, scope=scope)
    model.eval_logits = dense_layers(eval_net, train=False, scope=scope)
    logger.fprint('Defining loss and optimizer')
    model.loss = tf.losses.sparse_softmax_cross_entropy(model.labels_node, model.logits)
    model.loss = tf.reduce_mean(model.loss)
    model.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    model.eval_logits = tf.nn.softmax(model.eval_logits)
    return model

def create_mix_model(config,comp=True,opt=True):
    logger.fprint('Creating a MIX model')
    scope = 'slac_graph_'
    model = Model()
    model.slac_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='slac_ph')
    model.eval_slac_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='eval_slac_ph')
    net = model.slac_data_node
    eval_net = model.eval_slac_data_node
    model.bruker1_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='bruker1_ph')
    model.eval_bruker1_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='eval_bruker1_ph')
    model.bruker2_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='bruker2_ph')
    model.eval_bruker2_data_node = tf.placeholder(tf.float32, shape=[batch_size, 2048, 2048, 1], name='eval_bruker2_ph')
    model.labels_node = tf.placeholder(tf.int64, shape=batch_size, name='labels')
    net1 = model.bruker1_data_node
    eval_net1 = model.eval_bruker1_data_node
    net2 = model.bruker2_data_node
    eval_net2 = model.eval_bruker2_data_node
    input_channels=1

    model.inp_ph = {'Bruker1': model.bruker1_data_node, 'Bruker2': model.bruker2_data_node, 'labels': model.labels_node,
                    'SLAC': model.slac_data_node}
    model.eval_ph = {'Bruker1': model.eval_bruker1_data_node, 'Bruker2': model.eval_bruker2_data_node,
                     'SLAC': model.eval_slac_data_node}

    if use_conv:
        input_channels=4
        logger.fprint('Using conv to detect peaks')
        net = conv_input(net, train=True, conv_dict=config['conv_dict'])
        eval_net = conv_input(eval_net, train=False, conv_dict=config['conv_dict'])
        net1 = conv_input(net1, train=True, conv_dict=config['conv_dict'])
        eval_net1 = conv_input(eval_net1, train=False, conv_dict=config['conv_dict'])
        net2 = conv_input(net2, train=True, conv_dict=config['conv_dict'])
        eval_net2 = conv_input(eval_net2, train=False, conv_dict=config['conv_dict'])

    logger.fprint('Building conv graph for slac image')
    model.slac_conv_net = slac_conv_graph(net, scope=scope + '_conv_', input_channels=input_channels)
    model.eval_slac_conv_net = slac_conv_graph(eval_net, train=False, scope=scope + '_conv_', input_channels=input_channels)
    net = tf.layers.flatten(model.slac_conv_net, name='slac_flatten')
    eval_net = tf.layers.flatten(model.eval_slac_conv_net, name='slac_flatten')

    logger.fprint('Building conv graph for bruker image')
    model.bruker1_conv_net = bruker_conv_graph(net1, scope=scope + '_1_conv_', input_channels=input_channels)
    model.eval_bruker1_conv_net = bruker_conv_graph(eval_net1, train=False, scope=scope + '_1_conv_', input_channels=input_channels)
    model.bruker2_conv_net = bruker_conv_graph(net2, scope=scope + '_2_conv_', input_channels=input_channels)
    model.eval_bruker2_conv_net = bruker_conv_graph(eval_net2, train=False, scope=scope + '_2_conv_', input_channels=input_channels)
    net1 = tf.layers.flatten(model.bruker2_conv_net, name='bruker1_flatten')
    eval_net1 = tf.layers.flatten(model.eval_bruker1_conv_net, name='slac_flatten')
    net2 = tf.layers.flatten(model.bruker2_conv_net, name='bruker2_flatten')
    eval_net2 = tf.layers.flatten(model.eval_bruker2_conv_net, name='slac_flatten')

    net_b = tf.concat([net1, net2], axis=1)
    eval_net_b = tf.concat([eval_net1, eval_net2], axis=1)

    net_s = tf.expand_dims(net, 2)
    net_b = tf.expand_dims(net_b, 2)
    eval_net_s = tf.expand_dims(eval_net, 2)
    eval_net_b = tf.expand_dims(eval_net_b, 2)

    logger.fprint('net_b', net_b.get_shape())

    net = tf.concat([net_s, net_b], axis=2)
    eval_net = tf.concat([eval_net_s, eval_net_b], axis=2)
    net = tf.layers.max_pooling1d(net, pool_size=2, strides=1,data_format='channels_first')
    eval_net = tf.layers.max_pooling1d(eval_net, pool_size=2, strides=1, data_format='channels_first')
    net = tf.squeeze(net)
    eval_net = tf.squeeze(eval_net)

    if comp:
        logger.fprint('Building composition graph')
        model.comp_data_node = tf.placeholder(tf.float32, shape=[batch_size, 3])
        model.eval_comp_data_node = tf.placeholder(tf.float32, shape=[batch_size, 3])
        model.inp_ph['comp'] = model.comp_data_node
        model.eval_ph['comp'] = model.eval_comp_data_node
        model.comp_net = comp_graph(model.comp_data_node, scope=scope + '_comp_')
        model.eval_comp_net = comp_graph(model.eval_comp_data_node, train=False, scope=scope + '_comp_')
        net = tf.concat([net, model.comp_net], axis=1)
        eval_net = tf.concat([eval_net, model.eval_comp_net], axis=1)

    logger.fprint('Building classifier layers')
    model.logits = dense_layers(net, scope=scope)
    model.eval_logits = dense_layers(eval_net, train=False, scope=scope)
    logger.fprint('Defining loss and optimizer')
    model.loss = tf.losses.sparse_softmax_cross_entropy(model.labels_node, model.logits)
    model.loss = tf.reduce_mean(model.loss)
    if config['mix_conf']['opt']:
        #mp_loss = tf.losses.absolute_difference(tf.layers.flatten(net_s), tf.layers.flatten(net_b))
        net_s = tf.abs(tf.layers.flatten(net_s))
        net_b = tf.abs(tf.layers.flatten(net_b))
        sum_s = tf.reduce_sum(net_s, 1)
        sum_b = tf.reduce_sum(net_b, 1)
        cond_s = tf.equal(sum_s, tf.zeros(sum_s.get_shape()))
        cond_b = tf.equal(sum_b, tf.zeros(sum_b.get_shape()))
        cond = tf.logical_or(cond_s, cond_b)
        mp_loss = tf.where(cond, tf.zeros(cond.get_shape()), tf.abs(sum_s-sum_b))

        model.loss += tf.reduce_sum(mp_loss)
    model.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model.loss)
    model.eval_logits = tf.nn.softmax(model.eval_logits)
    return model


def save_model(config):
    sess = config['sess']
    saver = config['saver']
    config_ = {k:v for k,v in config.items() if not any([x in k for x in ['data','sess','labels', 'SLAC', 'Bruker', 'results', 'pred', 'stats', 'logger', 'test_data', 'train_data', 'sess', 'saver', 'save_path', 'test_rands', 'model_configs'] ])}
    save_folder = '/raid/dkj755/XRD/models'
    create_dir(save_folder)
    config['save_path'] = save_path = os.path.join(save_folder,md5sum(config_))
    #saver.save(sess, save_path)
    logger.fprint('saved to ',save_path, ' config is: ', str(config_))
    if not 'model_configs' in config: config['model_configs'] = {}
    config['model_configs'][config['cv_step']] = (config_, save_path, None)
    return

def restore_model(config, save_path):
    sess = config['sess']
    saver = config['saver']
    print 'model path is: %s'%save_path
    if os.path.exists(save_path+'.meta'):
        try:
            saver.restore(sess, save_path)
            logger.fprint('restored model from ', save_path)
            return True
        except:
            logger.fprint('failed to restore')
            return False
    else:
        logger.fprint('model does not exists')
        return False


def get_feed_dict(train_inds,config, model, train = True, input_type=None, data = None):
    feed_dict = {}
    if input_type is None: input_type = config['input_type']
    if train:
        if data is None: data = config['train_data']

        phs = model.inp_ph
        for k, v in phs.items():
            if k in ['SLAC', 'Bruker1', 'Bruker2']:
                k = k+'-'+input_type
            feed_dict[v] = data[k][train_inds, ...]
    else:
        if data is None: data = config['test_data']
        phs = model.eval_ph
        for k, v in phs.items():
            if k in ['SLAC', 'Bruker1', 'Bruker2']:
                k = k + '-'+input_type
            feed_dict[v] = data[k][train_inds, ...]

    return feed_dict

def eval_model(config,model, input_type=None, data=None):
        size = config['test_size']
        sess = config['sess']
        batch_size = config['batch_size']
        if size < batch_size:
            raise ValueError(
                'batch size for evals larger than dataset: %d' % size)
        predictions = np.ndarray(
                shape=(size, config['NUM_LABELS']), dtype=np.float32)
        for begin in range(0, size, batch_size):
                end = begin+batch_size
                if end > size:
                    test_inds = range(size - batch_size, size)
                else:
                    test_inds = range(begin, begin+batch_size)
                feed_dict = get_feed_dict(test_inds, config, model, train=False, input_type = input_type, data=data)

                if end <= size:
                    predictions[begin:end, :] = sess.run(model.eval_logits, feed_dict=feed_dict)
                else:
                    batch_predictions = sess.run(model.eval_logits, feed_dict=feed_dict)
                    predictions[-batch_size:, :] = batch_predictions
        return predictions

def get_accuracy(preds, labels, apply_argmax=True):
    if apply_argmax:
        return 100.0 * np.sum(np.argmax(preds, 1) == labels) / preds.shape[0]
    else:
        return 100.0 * np.sum(preds == labels) / preds.shape[0]


def get_predictions_all(config, model):
    logger.fprint('Getting predictions for all')
    if 'train_data' in config:  del config['train_data']
    if 'test_data' in config:   del config['test_data']
    config_ = None
    if not 'pred_all' in config:
        config['pred_all'] = {k:[] for k in config['test_types']}
        for k in config['pred_all']:
            config['pred_all'][k] = np.array([0 for _ in range(177)], dtype=np.int64)
    for input_type in config['pred_all']:
        cv_ratio = config['cv_ratio']
        if 'train_data' in config:  del config['train_data']
        if 'test_data' in config:   del config['test_data']
        total_splits = sum(cv_ratio) / cv_ratio[1]
        data = load_data(config, [input_type])
        accs = []
        for s in range(total_splits):
            test_inds = config['test_rands'][s]
            test_data = {}
            for k in data:
                test_data[k] = data[k][test_inds,...]
            config['test_data'] = test_data
            config['test_size'] = len(test_inds)
            config_, save_path, res = config['model_configs'][s]
            restore_model(config, save_path)

            config['pred_all'][input_type][test_inds] =  np.argmax(eval_model(config, model, input_type=input_type, data=test_data),1)
            acc = get_accuracy(config['pred_all'][input_type][test_inds], config['labels'][test_inds],
                                apply_argmax=False)
            accs.append(acc)
        if not 'stats' in config: config['stats'] = dict()
        config['stats'][input_type] = [np.mean(accs), np.std(accs)]

    if not config['model']=='MIX': return
    #test_inds = np.concatenate([test_inds, [177+x for x in test_inds], [2*177+x for x in test_inds]])
    #test_data = config['test_data']
    #comp_data = np.copy(test_data['comp'])

    #test_data['comp'] = np.concatenate((comp_data, comp_data, comp_data))
    if not 'MIXED-AUG' in config['pred_all']:
        config['labels-aug'] = np.concatenate([config['labels'], config['labels'], config['labels']])
        config['pred_all']['MIXED-AUG'] = {k:[] for k in config['test_types']}
        for k in config['pred_all']['MIXED-AUG']:
            config['pred_all']['MIXED-AUG'][k] = np.array([0 for _ in range(177*3)], dtype=np.int64)

    for input_type in config['test_types']:
        cv_ratio = config['cv_ratio']
        total_splits = sum(cv_ratio) / cv_ratio[1]
        total_data_points = config['total_data_points']
        if 'train_data' in config:  del config['train_data']
        if 'test_data' in config:   del config['test_data']

        data = load_data(config, [input_type])
        test_data = {}
        for s in range(total_splits):
            config_, save_path, res = config['model_configs'][s]
            restore_model(config, save_path)
            test_inds = config['test_rands'][s]
            test_data = {}
            slac_data = np.copy(data['SLAC' + '-' + input_type][test_inds,...])
            comp_data = data['comp'][test_inds,...]
            bruker1_data = np.copy(data['Bruker1' + '-' + input_type][test_inds,...])
            bruker2_data = np.copy(data['Bruker2' + '-' + input_type][test_inds,...])
            dumm_data = np.zeros((len(test_inds), 2048, 2048, 1), dtype=np.float32)
            test_data['SLAC' + '-' + input_type] = np.concatenate((slac_data, slac_data, dumm_data))
            test_data['Bruker1' + '-' + input_type] = np.concatenate((bruker1_data, dumm_data, bruker1_data))
            test_data['Bruker2' + '-' + input_type] = np.concatenate((bruker2_data, dumm_data, bruker2_data))
            test_data['comp'] = np.concatenate((comp_data, comp_data, comp_data))
            test_inds_aug = np.concatenate([test_inds, [177+x for x in test_inds], [2*177+x for x in test_inds]])
            config['test_data'] = test_data
            config['test_size'] = len(test_inds)
            config['pred_all']['MIXED-AUG'][input_type][test_inds_aug] = np.argmax(eval_model(config, model, input_type=input_type, data=test_data), 1)
        return

def analyze_save_predictions(config):
    logger.fprint('Analyzing accuracy for current config')
    labels = config['labels']
    config_ = {k:v for k,v in config.items() if not any([x in k for x in ['data','sess','labels', 'SLAC', 'Bruker', 'results', 'pred', 'logger', 'test_data', 'train_data', 'test_rands', 'saver', 'model_configs'] ])}
    acc_dict = {}
    if not 'results' in config: config['results'] = []
    for input_type in config['test_types']:
        if config['model'] =='MIX':
            acc1 = get_accuracy(config['pred_all']['MIXED-AUG'][input_type][:177], config['labels-aug'][:177], apply_argmax=False)
            acc2 = get_accuracy(config['pred_all']['MIXED-AUG'][input_type][177:177*2], config['labels-aug'][177:177*2], apply_argmax=False)
            acc3 = get_accuracy(config['pred_all']['MIXED-AUG'][input_type][177*2:], config['labels-aug'][177*2:], apply_argmax=False)
            logger.fprint('model: %s input_type:%s accuracy: %.3f %.3f %.3f' % (config['model'], input_type, acc1, acc2, acc3))
            acc = [acc1,acc2, acc3]
        else:
            acc = get_accuracy(config['pred_all'][input_type], labels, apply_argmax=False)
            logger.fprint('model: %s input_type:%s accuracy: %.3f'%(config['model'],input_type, acc))
        #config['results'].append((config_, input_type, acc))
        acc_dict[input_type] = (acc, config['stats'][input_type])
        if config['model'] == 'MIX':
            logger.fprint('predicted labels: ',config['pred_all']['MIXED-AUG'][input_type])
        else:
            logger.fprint('predicted labels: ', config['pred_all'][input_type])
        config['results'].append((config_, acc_dict))
        logger.fprint('actual labels: ', config['labels'])
        logger.fprint('mean and std: ', config['stats'][input_type])

def print_result_summary(config):
    logger.fprint('\n\nRESULTS:')
    if not config['results']: return
    results = config['results']
    if not results: return
    for k in results:
        logger.fprint(str(k))
    logger.fprint('\n\n')


def get_conv_features(config,model, data):
    pass


def data_aug(config):
    logger.fprint('Augmenting data')
    input_type = config['input_type']
    train_data = config['train_data']
    labels = np.copy(config['train_data']['labels'])
    slac_data = np.copy(train_data['SLAC'+'-'+input_type])
    bruker1_data = np.copy(train_data['Bruker1' + '-' + input_type])
    bruker2_data = np.copy(train_data['Bruker2' + '-' + input_type])
    comp_data = np.copy(train_data['comp'])
    dumm_data = np.zeros((labels.shape[0], 2048, 2048, 1), dtype=np.float32)

    train_data['SLAC'+'-'+input_type] = np.concatenate((slac_data, slac_data, dumm_data))
    train_data['Bruker1' + '-' + input_type] = np.concatenate((bruker1_data,dumm_data, bruker1_data))
    train_data['Bruker2' + '-' + input_type] = np.concatenate((bruker2_data, dumm_data, bruker2_data))
    train_data['comp'] = np.concatenate((comp_data, comp_data, comp_data))
    train_data['labels'] = np.concatenate((labels, labels, labels))
    randomize = np.arange(train_data['labels'].shape[0])
    np.random.shuffle(randomize)
    for k in train_data:
        train_data[k] = train_data[k][randomize,...]

def train_model(config, model):
    logger.fprint('Training model')
    saver = tf.train.Saver()
    config['saver'] = saver
    sess = tf.Session()
    config['sess'] = sess
    init = tf.global_variables_initializer()
    sess.run(init)
    num_epochs = config['num_epochs']
    train_size = config['train_size']
    test_size = config['test_size']
    best_acc = 0
    best_step = 0
    best_loss = 100000
    patience = config['patience']
    logger.fprint('Train size: %d test size: %d'%(train_size, test_size))
    for step in range(num_epochs):
        save_model(config)
        break
        start_time = time.time()
        total_loss = 0
        for iter in range(train_size//batch_size+1):
            if (iter+1)*batch_size > train_size:
                train_inds = range(train_size-batch_size, train_size)
            else:
                train_inds = range(iter*batch_size, (iter+1)*batch_size)
            feed_dict = get_feed_dict(train_inds,config, model)
            _, loss, _ = sess.run([model.logits, model.loss, model.optimizer], feed_dict=feed_dict)
            total_loss += loss
        loss = total_loss/(train_size//batch_size+1)
        eval_preds = eval_model(config, model)
        eval_acc = get_accuracy(eval_preds, config['test_data']['labels'])
        time_taken = time.time() - start_time
        logger.fprint('Epoch %d (time taken: %.1f seconds) training loss: %.4f eval acc: %.3f'%(step, time_taken, loss, eval_acc))
        if best_acc <= eval_acc and best_loss > loss:
            best_acc = eval_acc
            best_step = step
            best_loss = loss
            save_model(config)
        if best_step+patience < step and best_loss < loss: break
    return


def train_model_cv(config):
    tf.reset_default_graph()
    cv_ratio = config['cv_ratio']
    load_config(config)
    logger.fprint('\nPerforming cross validation %d:%d'%(cv_ratio[0], cv_ratio[1]))
    logger.fprint('\ncurrent config is ', {k:v for k,v in config.items() if not any([x in k for x in ['data','sess','labels', 'SLAC', 'Bruker', 'results', 'pred', 'logger', 'test_data', 'train_data'] ])})
    total_splits = sum(cv_ratio)/cv_ratio[1]
    np.random.seed(SEED)
    tf.set_random_seed(SEED)
    total_data_points = config['total_data_points']
    randomize = np.arange(total_data_points)
    np.random.shuffle(randomize)
    config['test_rands'] = {}
    if 'model_configs' in config:   del config['model_configs']
    if 'data' in config: del config['data']
    if 'train_data' in config: del config['train_data']
    if 'test_data' in config: del config['test_data']
    config['data'] = load_data(config, [config['input_type']])
    for s in range(total_splits):
        config['cv_step'] = s
        np.random.seed(SEED)
        tf.set_random_seed(SEED)
        tf.reset_default_graph()
        logger.fprint('\n CV step %d out of %d'%(s+1, total_splits))
        s_ind = int(1.*s/total_splits * total_data_points)
        e_ind = int(1.*(s+1)/total_splits * total_data_points)
        if s+1 == total_splits: e_ind = total_data_points
        if total_splits == total_data_points:
            s_ind = s
            e_ind = s+1
        test_rand = randomize[s_ind:e_ind]
        logger.fprint('Current test indices: ',test_rand)
        train_rand = [x for x in randomize if x not in test_rand]
        config['train_size'] = len(train_rand)
        config['test_size'] = len(test_rand)
        config['test_rands'][s] = test_rand
        train_data = {}
        test_data = {}
        data = config['data']
        if 'labels' not in config: config['labels'] = data['labels']
        for k in data.keys():
            if config['model'] == 'MIX':
                train_data[k] = np.copy(data[k][train_rand,...])
            else:
                train_data[k] = data[k][train_rand,...]
            if config['model'] == 'MIX':
                test_data[k] = np.copy(data[k][test_rand,...])
                del data[k]
            else:
                test_data[k] = data[k][test_rand,...]
        if config['model'] == 'MIX':    del config['data']
        config['train_data'] = train_data
        config['test_data'] = test_data
        if config['model'] =='comp':
            model = create_comp_model(config, comp=config['comp'])
        elif config['model'] =='SLAC':
            model = create_slac_model(config, comp=config['comp'])
        elif config['model'] =='Bruker':
            model = create_bruker_model(config, comp=config['comp'])
        elif config['model'] == 'CONCAT':
            model = create_concat_model(config, comp=config['comp'])
        elif config['model'] == 'MIX':
            model = create_mix_model(config, comp=config['comp'])
            if config['mix_conf']['data_aug']:
                data_aug(config)
        train_model(config, model)
    get_predictions_all(config, model)
    tf.reset_default_graph()
    analyze_save_predictions(config)
    logger.fprint('\n\n')
    print_result_summary(config)
