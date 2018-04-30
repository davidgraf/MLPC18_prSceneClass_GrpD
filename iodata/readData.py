'''
Created: 17.04.2018
@author:
description:
parameter:
'''

import os
import numpy as np
import random
import collections
from sklearn.naive_bayes import GaussianNB
from array import array

folds=['fold1']                #specify folds to read manually here
foldTypes=['train']         #select fold types

#Merge each scene into one big data matrix =>
def test(readFolds, cnt):
    gnb = GaussianNB()

    for l in readFolds['fold1_train']:
        for n in readFolds['fold1_train'][l]:
            print(n)
            gnb.fit(n, l)

#foldName= 'fold1' or 'fold2' or 'fold3' or 'fold4'
#foldtype= 'train' or 'test' or 'evaluate'
def readFold(foldName, foldtype, samplerate):

    data_path = os.getcwd() + '/iodata/data/mfcc_numpy'
    fold_path = os.getcwd() + '/iodata/data/evaluation_setup'

    print("reading from: " + fold_path +'/' + foldName + '_' + foldtype + '.txt')
    if foldtype == 'test':
        meta_data = np.loadtxt(fold_path +'/' + foldName + '_' + foldtype + '.txt', dtype='S100', delimiter='\t')
    else:
        meta_data = np.loadtxt(fold_path +'/' + foldName + '_' + foldtype + '.txt', dtype={'names': ('fileID', 'label'),
                                                                        'formats': ('S100', 'S100')}, delimiter='\t')

    rows = (len(meta_data) * int(501 * samplerate))
    feature_data = np.empty([rows, 60], dtype='float')
    label_data = np.empty(rows, dtype='S100')

    offset = 0
    for item in meta_data:          #for each label
        file_ID_full = item['fileID'] if not (foldtype == 'test') else item
        file_ID = str(file_ID_full).replace('audio/', '').replace('.wav', '').replace('b\'','').replace('\'','')
        file_path = str(file_ID) + ".npy"

        label = item['label'] if not (foldtype == 'test') else ''
        print('reading:' + str(file_ID_full) + ' - ' + str(label))

        matrix_tmp = np.load(str(data_path)+'/'+str(file_path)).transpose().astype(str)
        # sample according samplerate
        sampleIndexList = random.sample(range(0, 500), int(501 * samplerate))
        feature_data[offset:(offset + int(501 * samplerate)), 0:60] = matrix_tmp[sampleIndexList, :]
        label_data[offset:(offset + int(501 * samplerate))] = [label] * (int(501 * samplerate))
        offset = offset + int(501 * samplerate)

    print("done reading")

    return feature_data, label_data

#data = readFold('fold1', 'train', 0.01)
#print(data)
