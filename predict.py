'''
    python pridict.py <model_path>
'''
from keras.models import load_model
import numpy as np
import sys
import re

def generate_test_data():
    x = []
    num = 0
    with open('CIKM2017_testA/testA.txt') as f:
        for line in f:
            linebits = line.split(',') #__, label y, matrix x
            #y.append(float(linebits[1]))
            xstrlist = linebits[2].split(' ')
            xlist = [int(xstrlist[i]) for i in range(len(xstrlist))]
            xarray = np.array(xlist).reshape((15, 4, 101, 101))
            x.append(xarray[:,1,:,:])
            #x.append(xlist[:])
            num += 1
            if num == 100:
                x = np.array(x).reshape((100, 15, 101, 101))
                x = np.swapaxes(x, 1, 3)
                break
    return x

if __name__ == '__main__':

    '''loading model'''
    model_path = sys.argv[1]
    model = load_model(model_path)

    '''prediction'''
    test_X = generate_test_data()
    pred = model.predict(test_X, batch_size=10, verbose=0)

    '''
    for p in pred:
        m = re.search('(\d+(\.\d*)?|\.\d+)', str(p))
        print(m.group(0))
    '''
    with open('ans.csv', 'w') as f:
        for p in pred:
            m = re.search('(\d+(\.\d*)?|\.\d+)', str(p))
            f.writelines(m.group(0)+'\n')

    print('Result saved into ans.csv')
