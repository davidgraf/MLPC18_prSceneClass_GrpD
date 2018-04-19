'''
Created: 17.04.2018
@author:
description:
parameter:
'''

import numpy as np
import sklearn as sk
import json
import os

#Merge each scene into one big data matrix =>
def appendData():
    data = readData('/Users/Florian/Documents/DCASE2017_development_set/mfcc_numpy')
    whole_data = None
    i = 0
    for file_ID in data:
        i = i + 1
        print(data[file_ID].shape)
        print(i)
        if whole_data is None:
            whole_data = data[file_ID]
        else:
            whole_data = np.append(whole_data, data[file_ID], axis=1)

    print(whole_data.shape)

def readData(path):
    meta_data = np.loadtxt('data/meta.txt', dtype={'names': ('fileID', 'label', 'labelID'),
                                         'formats': ('S100', 'S100', 'S100')}, delimiter='\t')
    print(meta_data)
    data = {}
    whole_json_file = path + '/whole.npy'

    # if (os.path.exists(whole_json_file)):
    #     return np.load(whole_json_file).item()

    for item in meta_data:
        file_ID = item['fileID'].replace('audio/', '').replace('.wav', '')
        file_path = path + '/' + file_ID + '.npy'
        # label = item['label']
        # if label not in data:
        #     data[label] = {}

        print('Process file:' + file_path)
        #data[label][file_ID] = np.load(file_path)
        data[file_ID] = np.load(file_path)


    # print('Save file to:' + whole_json_file)
    # np.save(whole_json_file, data)

    return data

#appendData()
readData('/Users/Florian/Documents/DCASE2017_development_set/mfcc_numpy')