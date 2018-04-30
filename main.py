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
SAMPLERATE = 0.1

# ----------------------

# imports
from iodata.readData import readFold
from learning.classification import trainModel, testModel
from processing.featureEvaluation import featureClassCoerr
from processing.featureScaling import featureScale


# read training data
traindata_feature, traindata_labels = readFold('fold1', 'train', SAMPLERATE)

# read test data
testdata_feature, testdata_labels = readFold('fold1', 'evaluate', SAMPLERATE)

# split feature matrix and labels
#featureMatrixTrain = featureScale(traindata[...,:60])
featureMatrixTrain = featureScale(traindata_feature)
#labelsTrain = traindata[...,60:].ravel()
labelsTrain = traindata_labels.ravel()

#featureMatrixTest = featureScale(testdata[...,:60])
featureMatrixTest = testdata_feature
labelsTest = testdata_labels.ravel()

# data analysis
# ... analysis(data)

# preprocssing (feature scaling, feature evaluation, feature selection)
featureClassCoerr(featureMatrixTrain,labelsTrain,range(0,60))

# training
model, meanCrossVal = trainModel(featureMatrixTrain, labelsTrain, CLASSIFIER)

# testing
accuracy, precision, recall, f1 = testModel(model, featureMatrixTest, labelsTest)

















