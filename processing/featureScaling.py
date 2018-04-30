'''
Created: 30.04.2018
@author: davidgraf
description:
parameter:
'''

import sklearn.preprocessing as pp

def featureScale (featureMatrix):

    # standardization (zero mean, variance of one)
    stdScaler = pp.StandardScaler().fit(featureMatrix)
    featureMatrixScaled = stdScaler.transform(featureMatrix)

    # min-max scaler
    # minmaxScaler = pp.MinMaxScaler(copy=True, feature_range=(0, 1)).fit(featureMatrix)
    # featureMatrixScaled = minmaxScaler.transform(featureMatrix)

    return featureMatrixScaled
