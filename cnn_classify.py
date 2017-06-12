'''
    python cnn_classify.py <model_name>
'''
from __future__ import print_function

import sys
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, RMSprop, Adagrad
from keras.regularizers import l1, l2
from keras.layers.normalization import BatchNormalization
from keras.initializers import RandomNormal
from keras.utils import np_utils

model_name = sys.argv[1]

def data_generator(batch_size, split='train'):
    while 1:
        if split == 'train':
            f = open('CIKM2017_train/train_data.txt')
        elif split == 'valid':
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
            num += 1
            if num == batch_size:
                x = np.array(x).reshape((batch_size, 15, 101, 101))
                x = np.swapaxes(x, 1, 3)
                y = np.clip(np.array(y).astype(int), 0, 99)
                y = np_utils.to_categorical(y, 100)
                yield x, y
                x = []
                y = []
                num = 0
        f.close()

'''CNN model'''
model = Sequential()
model.add(Conv2D(128, (5, 5), padding='same', input_shape=(101, 101, 15)))
model.add(Activation('relu'))
model.add(Conv2D(64, (5, 5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
model.add(Conv2D(32, (1, 1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(32, (1, 1)))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))


model.add(Flatten())
model.add(Dense(100))
model.add(Activation('softmax'))

model.summary()

'''setting optimizer'''
model.compile(loss='categorical_crossentropy',
				optimizer='Adam',
				metrics=['accuracy'])

'''checkpoints'''
from keras.callbacks import ModelCheckpoint, EarlyStopping

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 5)
checkpoints = ModelCheckpoint(model_name+'.weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                        monitor='val_loss', save_best_only=True,
                        save_weights_only=False, mode='auto', period=1)

'''fit model'''
BATCH_SIZE = 100
history = model.fit_generator( data_generator(BATCH_SIZE, 'train'),
                               steps_per_epoch=90,
                               epochs=100,
                               validation_data=data_generator(BATCH_SIZE, 'valid'),
                               validation_steps=10,
                               callbacks=[early_stopping, checkpoints],
                               verbose=1)
'''
history = model.fit( X_train,
                     Y_train,
					 epochs=90, # epochs
					 shuffle=True, # shuffle
					 validation_split=0.1, # validation_split
					 verbose=1)
'''

'''Access the loss and accuracy in every epoch'''
loss	= history.history.get('loss')
acc 	= history.history.get('acc')
val_loss = history.history.get('val_loss')
val_acc = history.history.get('val_acc')

''' Visualize the loss and accuracy of both models'''
plt.figure(0)
plt.subplot(121)
plt.plot(range(len(loss)), loss,label='loss')
plt.plot(range(len(val_loss)), val_loss,label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.subplot(122)
plt.plot(range(len(acc)), acc,label='accuracy')
plt.plot(range(len(val_acc)), val_acc,label='Validation')
plt.title('Accuracy')
plt.savefig(model_name + '.png',dpi=300,format='png')
plt.close()
print('Result saved into' + model_name + '.png')
