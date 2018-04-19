'''
Created: 17.04.2018
@author:
description:
parameter:
'''

import numpy as np
import sklearn as sk

file_path = '/Users/Florian/Documents/DCASE2017_development_set/mfcc_numpy'

def readData(path):

    # ... read features and labels from files and return data structure (already split to 4-folds?)
    data = np.load(file_path & '/mfcc_numpy/a001_0_10.npy')
    print(data)
    data1 = np.load('/Volumes/GoogleDrive/My Drive/WinStudium/MSc/ Machine Learning and Pattern Classification/DCASE2017_development_set/mfcc_numpy/a001_10_20.npy')
    print(data1)
    return None




