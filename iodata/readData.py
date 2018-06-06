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


def _readData(meta_data, samplerate, data_path, blabel_data=True):
    rows = (len(meta_data) * int(501 * samplerate))
    feature_data = np.empty([rows, 60], dtype='float')
    label_data = np.empty(rows, dtype='S100') if blabel_data else None
    file_name = np.empty(rows, dtype='S100')

    offset = 0
    for item in meta_data:          #for each label
        file_ID_full = item['fileID'] if blabel_data else item
        file_ID = str(file_ID_full).replace('audio/', '').replace('.wav', '').replace('b\'','').replace('\'','')
        file_path = str(file_ID) + ".npy"

        label = item['label'] if blabel_data else None
        #print('reading:' + str(file_ID_full) + ' - ' + str(label))

        matrix_tmp = np.load(str(data_path)+'/'+str(file_path)).transpose().astype(str)
        # sample according samplerate
        sampleIndexList = random.sample(range(0, 500), int(501 * samplerate)) if samplerate < 1.0 else range(0, 500)
        feature_data[offset:(offset + int(501 * samplerate)), 0:60] = matrix_tmp[sampleIndexList, :] if samplerate < 1.0 else matrix_tmp
        if blabel_data:
            label_data[offset:(offset + int(501 * samplerate))] = [label] * (int(501 * samplerate))

        file_name[offset:(offset + int(501 * samplerate))] = [file_ID] * (int(501 * samplerate))
        offset = offset + int(501 * samplerate)

    return feature_data, label_data, file_name


#foldName= 'fold1' or 'fold2' or 'fold3' or 'fold4'
#foldtype= 'train' or 'test' or 'evaluate'
def readFold(foldName, foldtype, samplerate, data_folder='/iodata/data/'):

    data_path = os.getcwd() + data_folder + 'mfcc_numpy'
    fold_path = os.getcwd() + data_folder + 'evaluation_setup'

    print("reading from: " + fold_path +'/' + foldName + '_' + foldtype + '.txt')
    if foldtype == 'test':
        meta_data = np.loadtxt(fold_path +'/' + foldName + '_' + foldtype + '.txt', dtype='S100', delimiter='\t')
    elif foldtype == 'whole':
        meta_data = np.loadtxt(os.getcwd() + data_folder + '/' + 'meta.txt',dtype={'names': ('fileID', 'label'),
                                      'formats': ('S100', 'S100')}, delimiter='\t')
        print(meta_data, len(meta_data))
        # meta_data_train = np.loadtxt(fold_path + '/' + foldName + '_' + 'train' + '.txt',
        #                        dtype={'names': ('fileID', 'label'),
        #                               'formats': ('S100', 'S100')}, delimiter='\t')
        #
        # meta_data_eval = np.loadtxt(fold_path + '/' + foldName + '_' + 'evaluate' + '.txt',
        #                        dtype={'names': ('fileID', 'label'),
        #                               'formats': ('S100', 'S100')}, delimiter='\t')
        #
        # meta_data = np.append(meta_data_eval, meta_data_train)
        #print(meta_data, len(meta_data), len(meta_data_train), len(meta_data_eval))
    else:
        meta_data = np.loadtxt(fold_path +'/' + foldName + '_' + foldtype + '.txt', dtype={'names': ('fileID', 'label'),
                                                                        'formats': ('S100', 'S100')}, delimiter='\t')

    feature_data, label_data, file_name = _readData(meta_data, samplerate, data_path=data_path, blabel_data=True)

    print("done reading")

    return feature_data, label_data, file_name


def readPredictData(data_folder='/iodata/data/mfcc_d'):
    print(os.getcwd() + data_folder)
    data_path = os.getcwd() + data_folder
    files = os.listdir(os.getcwd() + data_folder)
    meta_data = [str(f).replace('.npy', '') for f in sorted(files, key = lambda k: int(k.replace('.npy', '')))]
    #print(meta_data)
    feature_data, label_data, file_name = _readData(meta_data, 1.0, data_path=data_path, blabel_data=False)

    return feature_data, file_name

#data, label, files = readFold('whole', 'whole', 1, data_folder='/data/')
# print(len(data), data, files)

#print(readPredictData(1, data_folder='/data/mfcc_d/'))