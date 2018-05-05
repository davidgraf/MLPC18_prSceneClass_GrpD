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
CLASSIFIER = 'NaiveBayes'

# for sampling 0.1 means only 10%
SAMPLERATE = 0.01

# ----------------------

# imports
from iodata.readData import readFold
from learning.classification import trainModel, testModel
from processing.featureEvaluation import featureClassCoerr
from processing.featureScaling import featureScale
import time


# read training data
traindata_feature, traindata_labels = readFold('fold1', 'train', SAMPLERATE)

# read test data
testdata_feature, testdata_labels = readFold('fold1', 'evaluate', SAMPLERATE)

# preprocssing (feature scaling, feature evaluation, feature selection)
# featureClassCoerr(featureMatrixTrain,labelsTrain,range(0,60))

# scale train data
featureMatrixTrain, scaler = featureScale(traindata_feature[:])
labelsTrain = traindata_labels

# scale test data according train values
featureMatrixTest = scaler.transform(testdata_feature)
labelsTest = testdata_labels

# data analysis
# ... analysis(data)

timeStart = time.time()

# training
model, meanCrossVal = trainModel(featureMatrixTrain, labelsTrain, CLASSIFIER)

timeStartPredict=time.time();

# testing
accuracy, precision, recall, f1 = testModel(model, featureMatrixTest, labelsTest)

print "Training time (sec.)",(timeStartPredict-timeStart)
print "Prediction time (sec.)",(time.time()-timeStartPredict)


















