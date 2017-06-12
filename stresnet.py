'''
    python stresnet.py <model_name>

    Model architecture from ST-ResNet: Deep Spatio-temporal Residual Networks
    https://github.com/lucktroy/DeepST
    Junbo Zhang, Yu Zheng, Dekang Qi.
    Deep Spatio-Temporal Residual Networks for Citywide Crowd Flows Prediction.
    In AAAI 2017.
'''
from __future__ import print_function
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Reshape,
    Lambda
)
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import sys

import matplotlib.pyplot as plt

def data_generator(batch_size, split='train'):
    while 1:
        if split == 'train':
            #f = open('data_new/CIKM2017_train/train_1000.txt') #1000
            f = open('CIKM2017_train/train_data.txt')
        elif split == 'valid':
            #f = open('data_new/CIKM2017_train/small_file_2100.txt') #100
            f = open('CIKM2017_train/valid_data.txt')
        x = []
        y = []
        num = 0
        for line in f:
            linebits = line.split(',') #__, label y, matrix x
            y.append(float(linebits[1]))
            xstrlist = linebits[2].split(' ')
            xlist = [int(xstrlist[i]) for i in range(len(xstrlist))]
            xarray = np.array(xlist).reshape((15, 4, 101, 101))
            x.append(xarray[:,1,:,:])
            #x.append(xlist[:])
            num += 1
            if num == batch_size:
                x = np.array(x).reshape((batch_size, 15, 101, 101))
                x = np.swapaxes(x, 1, 3)
                #y = np.clip(np.array(y).astype(int), 0, 99)
                #y = np_utils.to_categorical(y, 100)
                y = np.array(y)
                yield x, y
                x = []
                y = []
                num = 0
        f.close()

def plot_training(model_name, history):
    '''Access the loss and accuracy in every epoch'''
    loss	= history.history.get('loss')
    #acc 	= history.history.get('acc')
    val_loss = history.history.get('val_loss')
    #val_acc = history.history.get('val_acc')

    ''' Visualize the loss and accuracy of both models'''
    plt.figure(0)
    #plt.subplot(121)
    plt.plot(range(len(loss)), loss,label='loss')
    plt.plot(range(len(val_loss)), val_loss,label='Validation')
    plt.title('Loss')
    plt.legend(loc='upper left')
    #plt.subplot(122)
    #plt.plot(range(len(acc)), acc,label='accuracy')
    #plt.plot(range(len(val_acc)), val_acc,label='Validation')
    #plt.title('Accuracy')
    plt.savefig(model_name+'.png',dpi=300,format='png')
    plt.close()
    print('Result saved into '+model_name+'.png')


def _shortcut(input, residual):
    return merge([input, residual], mode='sum')


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, border_mode="same")(activation)
    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3)(residual)
        return _shortcut(input, residual)
    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input
    return f

def stresnet(conf=(15, 4, 101, 101), nb_residual_unit=3):
    '''
    conf = (time_span, height_span, map_height, map_width)
    '''

    time_span, height_span, map_height, map_width = conf
    input = Input(shape=(map_height, map_width, time_span * height_span))

    # Conv1
    conv1 = Convolution2D(
        nb_filter=64, nb_row=3, nb_col=3, border_mode="same")(input)
    # [nb_residual_unit] Residual Units
    residual_output = ResUnits(_residual_unit, nb_filter=64,
                      repetations=nb_residual_unit)(conv1)
    # Conv2
    activation = Activation('relu')(residual_output)
    conv2 = Convolution2D(
        nb_filter=1, nb_row=3, nb_col=3, border_mode="same")(activation)

    output = Activation('tanh')(conv2)
    output = Flatten()(output)
    output = Dense(1)(output)
    #main_output = Lambda(lambda x: x[:,50,50,:])(main_output)
    model = Model(input=input, output=output)

    model.compile(loss='mse', optimizer='Adam')

    return model

if __name__ == '__main__':
    model_name = sys.argv[1]
    model = stresnet(conf=(15, 1, 101, 101), nb_residual_unit=2)

    '''set checkpoints'''
    #early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
    tensorboard = TensorBoard(log_dir='./logs/', histogram_freq=0)
    checkpoints = ModelCheckpoint('./model/weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                            monitor='val_loss', save_best_only=True,
                            save_weights_only=False, mode='auto', period=1)

    '''Fit model'''
    BATCH_SIZE = 100
    history = model.fit_generator( data_generator(BATCH_SIZE, 'train'),
                                   steps_per_epoch=90,
                                   epochs=90,
                                   validation_data=data_generator(BATCH_SIZE, 'valid'),
                                   validation_steps=10,
                                   callbacks=[tensorboard, checkpoints],
                                   verbose=1)

    plot_training(model_name, history)

    '''saving model'''
    model.save(model_name + '.h5')
