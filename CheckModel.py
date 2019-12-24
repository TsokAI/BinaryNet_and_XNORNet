if __name__ == '__main__':
    import os
    gpu_use = 2
    print('GPU use: {}'.format(gpu_use))
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)

    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

import numpy as np
from CustomOps.customOps import SetSession

# Call this first here, to make sure that Tensorflow registers our custom ops properly
SetSession()

from keras.models import Model, load_model
from keras.layers import Dense, Flatten, Input
from keras.optimizers import Adam
from keras.datasets import mnist, cifar100, cifar10
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from CompositeLayers.ConvBNReluLayer import ConvBNReluLayer
from CompositeLayers.BinaryNetConvBNReluLayer import BinaryNetConvBNReluLayer, BinaryNetActivation
from CustomLayers.CustomLayersDictionary import customLayersDictionary
from CompositeLayers.XNORConvLayer import XNORConvBNReluLayer, BNXNORConvReluLayer
from NetworkParameters import NetworkParameters
from CustomLayers.CustomLayersDictionary import customLayerCallbacks

np.random.seed(1337)  # for reproducibility


model_path = './savedModels/best_model_XNor.hdf5'
model = load_model(filepath=model_path, custom_objects=customLayersDictionary)
print(model.summary())

for level_id in range(len(model.layers)):
    layer = model.layers[level_id]
    layer_type = layer.__class__.__name__
    print('Layer num: {} Layer name: {} Layer type: {}'.format(level_id, layer.name, layer_type))

    if layer_type == 'Conv2D':
        w = layer.get_weights()[0]
        print(layer.get_weights())

    elif layer_type == 'BatchNormalization':
        print(layer.get_weights())

    elif layer_type == 'Activation':
        print(layer.get_weights())

    elif layer_type == 'XNORNetConv2D':
        print(layer.get_weights())

    elif layer_type == 'BinaryNetConv2D':
        print(layer.get_weights())

    elif layer_type == 'Dense':
        print(layer.get_weights())
