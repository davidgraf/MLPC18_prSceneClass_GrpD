from iodata.readData import readPredictData, readFold
from iodata.writeData import writeToCSV
import numpy as np
from learning.classification import trainModel, testModel, predict_per_file
from processing.featureEvaluation import featureClassCoerr
from processing.featureScaling import featureScale
import time
import operator

SAMPLERATE = 0.1
overallAccuracy = 0


class Fold:
    features = np.array(None)
    labels = np.array(None)
    file_names = np.array(None)


class ClassifierResults:
    def __init__(self, classifier, setting):
        self.classifier = classifier
        self.setting = setting
        self.results = {}


settings = {
    'RandomForest':[
        {
           'n_estimators': 100,
           'criterion': 'gini'
        },
    ],
    'SVM': [
       {
           'kernel': 'rbf',
       },
    ],
    'NeuroNet':[
        {
            'hidden_layer_sizes': (100,),
            'solver': 'adam',
            'learning_rate_init': 0.0001
        },
    ]
}

train = Fold()

train.features, train.labels, train.file_names = readFold("whole", "whole", SAMPLERATE)

test = Fold()

test.features, test.file_names = readPredictData()

classifier_results = []
for classifier in settings.keys():
    print('************************************' + classifier + '************************************')
    for setting in settings[classifier]:
        classifier_cur_res = ClassifierResults(classifier, setting)

        print('********NewSetting********')
        print('Settings: ' + str(setting))
        overallAccuracy = 0
        timeStart = time.time()

        t_features = featureScale(train.features)
        t_labels = train.labels.ravel()

        e_features = test.features
        featureMatrixTest = featureScale(e_features)
        #labelsTest = e_labels.ravel()
        fileNamesTest = test.file_names.ravel()

        # training
        model, meanCrossVal = trainModel(t_features[0], t_labels, classifier, setting)

        # prediction
        predicted_dict = predict_per_file(model, featureMatrixTest[0], None, fileNamesTest)

        for file in predicted_dict:
            classifier_cur_res.results[file] = predicted_dict[file]['predicted']

        classifier_results.append(classifier_cur_res)

        print("Prediction Time (sec.)", (time.time() - timeStart))


predicted = {}

toCSV = []
'''For each classifier -> setting -> file: count the prediction'''

for c in classifier_results:
    for r in c.results:
        if r not in predicted:
            predicted[r] = {}
            for p in c.results[r]:
                if p not in predicted[r]:
                    predicted[r][p] = 1
                else:
                    predicted[r][p] += 1

        else:
            for p in c.results[r]:
                if p not in predicted[r]:
                    predicted[r][p] = 1
                else:
                    predicted[r][p] += 1

ensemble_t = 0
ensemble_f = 0

'''Order the predictions by count and select the highest one, then compare it to the actual label for this file'''
for p in predicted:
    toCSV.append([p, list(sorted(predicted[p],key=predicted[p].get, reverse=True))[0]])
    # if list(sorted(predicted[p],key=predicted[p].get, reverse=True))[0] == actual[p]:
    #     ensemble_t += 1
    # else:
    #     ensemble_f += 1


writeToCSV(toCSV,'prediction.csv')
print('Predicting Done!')
#print("Ensemble accurracy: #true: " + str(ensemble_t) + ' #false: ' + str(ensemble_f) + ' acc: ' + str(ensemble_t/float(ensemble_t+ensemble_f)))
