'''
Created: 17.04.2018
@author:
description:
parameter:
'''

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

def readData(path):
    data_path = 'D:/OneDrive - Johannes Kepler Universität Linz/Uni/Informatik/Master/Semester1/ML/Project/mfcc_numpy'      #specify path where .npy files are stored
    fold_path = 'D:/OneDrive - Johannes Kepler Universität Linz/Uni/Informatik/Master/Semester1/ML/Project/TUT-acoustic-scenes-2017-development/evaluation_setup'      #specify path where fold files are stored
    readFolds = {}
    for f in folds:
        for fType in foldTypes:
            print("reading from: "+fold_path+'/'+f+'_'+fType+'.txt')
            meta_data = np.loadtxt(fold_path+'/'+f+'_'+fType+'.txt', dtype={'names': ('fileID', 'label'),
                                                                            'formats': ('S100', 'S100')}, delimiter='\t')
            foldData = {}                   #the current fold: label: all matrizes
            cnt = collections.Counter([i for j in meta_data for i in j])        #gives us the number of occurrences for each label
            for item in meta_data:          #for each label
                file_ID = str(item['fileID']).replace('audio/', '').replace('.wav', '').replace('b\'','').replace('\'','')
                file_path = str(file_ID) + ".npy"
                label = item['label']
                if label not in foldData:       #add labels to the dictionary
                    print("reading: " +str(label))
                foldData.setdefault(label, []).append(np.load(str(data_path)+'/'+str(file_path)))
            readFolds[f+'_'+fType] = foldData
    print("done reading")
    test(readFolds, cnt)
readData('/Users/Florian/Documents/DCASE2017_development_set/mfcc_numpy')
