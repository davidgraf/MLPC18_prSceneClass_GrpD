from iodata.readData import readFold
import numpy as np
from learning.classification import trainModel, testModel
from processing.featureEvaluation import featureClassCoerr
from processing.featureScaling import featureScale
import time


folds = []

SAMPLERATE = 0.1
overallAccuracy = 0


class Fold:
    features = np.array(None)
    labels = np.array(None)

class FoldData:
    labels_train = None
    feature_matrix_train = None

#read the folds and store them for all iterations


folds = [Fold() for j in range (4)]
folds[0].features, folds[0].labels = readFold("fold1", "train", SAMPLERATE)
folds[1].features, folds[1].labels = readFold("fold2", "train", SAMPLERATE)
folds[2].features, folds[2].labels = readFold("fold3", "train", SAMPLERATE)
folds[3].features, folds[3].labels = readFold("fold4", "train", SAMPLERATE)

eval_folds = [Fold() for j in range(4)]
eval_folds[0].features, eval_folds[0].labels = readFold("fold1", "evaluate", SAMPLERATE)
eval_folds[1].features, eval_folds[1].labels = readFold("fold2", "evaluate", SAMPLERATE)
eval_folds[2].features, eval_folds[2].labels = readFold("fold3", "evaluate", SAMPLERATE)
eval_folds[3].features, eval_folds[3].labels = readFold("fold4", "evaluate", SAMPLERATE)

timeStart = time.time()

for i in range(4):              #4 folds

    trainData = [FoldData() for j in range (3)]

    #3 training folds

    trainData[0].feature_matrix_train = featureScale(folds[i].features)
    trainData[0].labels_train= folds[i].labels.ravel()
    trainData[1].feature_matrix_train = featureScale(folds[(i+1)%4].features)
    trainData[1].labels_train = folds[(i+1)%4].labels.ravel()
    trainData[2].feature_matrix_train = featureScale(folds[(i+2)%4].features)
    trainData[2].labels_train = folds[(i+2)%4].labels.ravel()

    featureMatrixTrain = np.append(trainData[0].feature_matrix_train[0], trainData[1].feature_matrix_train[0], axis = 0)
    featureMatrixTrain = np.append(featureMatrixTrain, trainData[2].feature_matrix_train[0], axis = 0)
    labelsTrain = np.append(trainData[0].labels_train, trainData[1].labels_train, axis=0)
    labelsTrain = np.append(labelsTrain, trainData[2].labels_train, axis=0)

    #1 evaluation fold

    e_features = eval_folds[(i+3)%4].features
    e_labels = eval_folds[(i+3)%4].labels

    featureMatrixTest = featureScale(e_features)
    labelsTest = e_labels.ravel()

    #featureClassCoerr(featureMatrixTrain, labelsTrain, range(0,60))

# training
    model, meanCrossVal = trainModel(featureMatrixTrain, labelsTrain, 'NaiveBayes')

# testing
    accuracy, precision, recall, f1 = testModel(model, featureMatrixTest[0], labelsTest)
    overallAccuracy += accuracy

print(overallAccuracy/4)

print "Train/Prediction Time (sec.)",(time.time()-timeStart)