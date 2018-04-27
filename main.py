'''
Created: 17.04.2018
@author: davidgraf
description: main application to run
parameter:
'''

# ---------------------
# Konfiguration

#DATA_DIR = "C:/Temp/DCASE2017_development_set"

# 'SVM' or 'DecisionTree' or 'RandomForest' or 'GaussianProcess' or 'AdaBoost' or 'NeuroNet' or 'NaiveBayes'
CLASSIFIER = 'SVM'

# for sampling 0.1 means only 10%
SAMPLERATE = 0.01

# ----------------------

# imports
from iodata.readData import readFold
from learning.classification import trainModel, testModel


# read training data
traindata = readFold('fold1', 'train', SAMPLERATE)

# read test data
testdata = readFold('fold1', 'evaluate', SAMPLERATE)

# split feature matrix and labels
featureMatrixTrain = traindata[...,:60]
labelsTrain = traindata[...,60:].ravel()

featureMatrixTest = testdata[...,:60]
labelsTest = testdata[...,60:].ravel()

# data analysis
# ... analysis(data)

# preprocssing (feature scaling, feature evaluation, feature selection)
# ...

# training
model, meanCrossVal = trainModel(featureMatrixTrain, labelsTrain, CLASSIFIER)

# testing
accuracy, precision, recall, f1 = testModel(model, featureMatrixTest, labelsTest)

















