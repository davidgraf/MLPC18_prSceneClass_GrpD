'''
Created: 17.04.2018
@author: davidgraf
description:
'''

from sklearn import svm
from sklearn import tree
from sklearn import gaussian_process
from sklearn import ensemble
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as scikitm


def SVMclassifierTrain(featureMatrix, labelMatrix):
    clf = svm.SVC()
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    #scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    #print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, None # scores.mean()


def classifierPredict(featureMatrix, model):
    return model.predict(featureMatrix)


def DecisionTreeTrain(featureMatrix, labelMatrix):
    clf = tree.DecisionTreeClassifier()
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    #scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    #print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, None #scores.mean()


def GaussianProcessTrain(featureMatrix, labelMatrix):
    clf = gaussian_process.GaussianProcessClassifier()
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    #scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    #print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, None #scores.mean()

def GaussianNaiveBayes(featureMatrix, labelMatrix):
    clf = GaussianNB()
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    #scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    #print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, None #scores.mean()


def AdaBoostTrain(featureMatrix, labelMatrix):
    clf = ensemble.AdaBoostClassifier()
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    #scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    #print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, None #scores.mean()


def RandomForestTrain(featureMatrix, labelMatrix):
    clf = ensemble.RandomForestClassifier(criterion="entropy")
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    #scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    #print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, None #scores.mean()


def NeuroNetTrain(featureMatrix, labelMatrix):
    clf = MLPClassifier(solver='adam', max_iter=400, hidden_layer_sizes=(25, 2))
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    #scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    #print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, None #scores.mean()

def trainModel(features, labels, classifier):
    # train classifier and apply cross validation
    print "------------------------- Train classifier -------------------------"

    if classifier == 'SVM':
        model, meanScore = SVMclassifierTrain(features, labels)
    elif classifier == 'DecisionTree':
        model, meanScore = DecisionTreeTrain(features, labels)
    elif classifier == 'RandomForest':
        model, meanScore = RandomForestTrain(features, labels)
    elif classifier == 'GaussianProcess':
        model, meanScore = GaussianProcessTrain(features, labels)
    elif classifier == 'NaiveBayes':
        model, meanScore = GaussianNaiveBayes(features, labels)
    elif classifier == 'AdaBoost':
        model, meanScore = AdaBoostTrain(features, labels)
    elif classifier == 'NeuroNet':
        model, meanScore = NeuroNetTrain(features, labels)
    else:
        print "No available classifier selected - please choose classification algorithm!"

    print "Train " + classifier + " on " + str(len(labels)) + " samples"

    return model, meanScore

def testModel(model, features, labels):
    # apply classifier on different dataset
    print "------------------------- Test classifier -------------------------"
    print "Test classifier on " + str(len(labels)) + " samples"
    predictedTest = classifierPredict(features, model)
    accuracy = scikitm.accuracy_score(labels, predictedTest)
    #precision = scikitm.precision_score(labels, predictedTest)
    #recall = scikitm.recall_score(labels, predictedTest)
    #f1 = scikitm.f1_score(labels, predictedTest)
    print "Accuracy (Testset): " + str(accuracy)
    #print "Precision (Testset): " + str(precision)
    #print "Recall (Testset): " + str(recall)
    #print "F1-Score (Testset): " + str(f1)

    return accuracy, None, None, None
