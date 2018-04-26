'''
Created: 17.04.2018
@author:
description:
parameter:
'''

import os
import numpy as np
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
def readFold(foldName, foldtype):
    #data_fold_dict_file = 'data_fold_dict.npy'
    #
    # if (os.path.exists('data/' + data_fold_dict_file)):
    #     return np.load('data/' + data_fold_dict_file).item()

    data_path = 'data/mfcc_numpy'
    fold_path = 'data/evaluation_setup'
    fold_data_dict = {}
    print("reading from: " + fold_path +'/' + foldName + '_' + foldtype + '.txt')
    cols = 61
    if foldtype == 'test':
        meta_data = np.loadtxt(fold_path +'/' + foldName + '_' + foldtype + '.txt', dtype='S100', delimiter='\t')
        cols = 60
    else:
        meta_data = np.loadtxt(fold_path +'/' + foldName + '_' + foldtype + '.txt', dtype={'names': ('fileID', 'label'),
                                                                        'formats': ('S100', 'S100')}, delimiter='\t')

    foldData = np.empty([(len(meta_data) * 501), cols],
                        dtype='S100')
    #cnt = collections.Counter([i for j in meta_data for i in j])        #gives us the number of occurrences for each label

    offset = 0
    for item in meta_data:          #for each label
        file_ID_full = item['fileID'] if cols == 61 else item
        file_ID = str(file_ID_full).replace('audio/', '').replace('.wav', '').replace('b\'','').replace('\'','')
        file_path = str(file_ID) + ".npy"

        label = item['label'] if cols == 61 else ''

        print('reading:' + str(file_ID_full) + ' - ' + str(label))
        matrix = np.load(str(data_path)+'/'+str(file_path)).transpose().astype(str)
        matrix = np.concatenate((matrix, [[label]]* len(matrix)), axis=1) if cols == 61 else matrix
            #.append(matrix, [[label]]* len(matrix), axis=1) if cols == 61 else matrix

        foldData[offset:(offset + 501), 0:cols] = matrix
        offset = offset + 501

    #fold_data_dict[foldName + '_' + foldtype] = foldData
    #print('saving to ' + 'data/folds/' + f + '_' + foldtype)
    #np.save('data/folds/' + f + '_' + foldtype, foldData.transpose())

    print("done reading")

    #np.save('data/'+ data_fold_dict_file, fold_data_dict)
    #print('done')
    return foldData

data = readFold('fold1', 'train')
print(data)
