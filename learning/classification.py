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
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, scores.mean()


def classifierPredict(featureMatrix, model):
    return model.predict(featureMatrix)


def DecisionTreeTrain(featureMatrix, labelMatrix):
    clf = tree.DecisionTreeClassifier()
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, scores.mean()


def GaussianProcessTrain(featureMatrix, labelMatrix):
    clf = gaussian_process.GaussianProcessClassifier()
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, scores.mean()

def GaussianNaiveBayes(featureMatrix, labelMatrix):
    clf = GaussianNB()
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, scores.mean()


def AdaBoostTrain(featureMatrix, labelMatrix):
    clf = ensemble.AdaBoostClassifier()
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, scores.mean()


def RandomForestTrain(featureMatrix, labelMatrix):
    clf = ensemble.RandomForestClassifier(criterion="entropy")
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, scores.mean()


def NeuroNetTrain(featureMatrix, labelMatrix):
    clf = MLPClassifier(solver='adam', max_iter=400, hidden_layer_sizes=(25, 2))
    print clf.fit(featureMatrix, labelMatrix)

    # cross validation
    scores = cross_val_score(clf, featureMatrix, labelMatrix, cv=5)
    print("Accuracy (Cross-V): %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return clf, scores.mean()

def trainModel(features, labels, events):
    # train classifier and apply cross validation
    print "------------------------- Train classifier -------------------------"
    print "Train on " + events + " Nr of Teeets: " + str(len(labels))
    model, meanScore = SVMclassifierTrain(features, labels)
    #model, meanScore = AdaBoostTrain(features, labels)

    return model, meanScore

def testModel(model, features, labels, events):
    # apply classifier on different dataset
    print "------------------------- Test classifier -------------------------"
    print "Test on " + events + " Nr of Teeets: " + str(len(labels))
    predictedTest = classifierPredict(features, model)
    accuracy = scikitm.accuracy_score(labels, predictedTest)
    precision = scikitm.precision_score(labels, predictedTest)
    recall = scikitm.recall_score(labels, predictedTest)
    f1 = scikitm.f1_score(labels, predictedTest)
    print "Accuracy (Testset): " + str(accuracy)
    print "Precision (Testset): " + str(precision)
    print "Recall (Testset): " + str(recall)
    print "F1-Score (Testset): " + str(f1)

    return accuracy, precision, recall, f1
