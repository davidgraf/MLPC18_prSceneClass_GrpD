from iodata.readData import readFold
import numpy as np
from learning.classification import trainModel, testModel, testModel_per_file
from processing.featureEvaluation import featureClassCoerr
from processing.featureScaling import featureScale
import time
import operator


folds = []

SAMPLERATE = 0.1
overallAccuracy = 0


class Fold:
    features = np.array(None)
    labels = np.array(None)
    file_names = np.array(None)

# class FoldData:
#     labels_train = None
#     feature_matrix_train = None

#read the folds and store them for all iterations


folds = [Fold() for j in range (4)]
folds[0].features, folds[0].labels, folds[0].file_names = readFold("fold1", "train", SAMPLERATE)
folds[1].features, folds[1].labels, folds[1].file_names = readFold("fold2", "train", SAMPLERATE)
folds[2].features, folds[2].labels, folds[2].file_names = readFold("fold3", "train", SAMPLERATE)
folds[3].features, folds[3].labels, folds[3].file_names = readFold("fold4", "train", SAMPLERATE)

eval_folds = [Fold() for j in range(4)]
eval_folds[0].features, eval_folds[0].labels, eval_folds[0].file_names = readFold("fold1", "evaluate", SAMPLERATE)
eval_folds[1].features, eval_folds[1].labels, eval_folds[1].file_names = readFold("fold2", "evaluate", SAMPLERATE)
eval_folds[2].features, eval_folds[2].labels, eval_folds[2].file_names = readFold("fold3", "evaluate", SAMPLERATE)
eval_folds[3].features, eval_folds[3].labels, eval_folds[3].file_names = readFold("fold4", "evaluate", SAMPLERATE)

settings = {
    'NaiveBayes': [{

    }],
    'RandomForest':[
        # {
        #    'n_estimators': 5,
        #    'criterion': 'entropy'
        # },
        # {
        #    'n_estimators': 1,
        #    'criterion': 'entropy'
        # },
        # {
        #    'n_estimators': 5,
        #    'criterion': 'entropy'
        # },
        # {
        #    'n_estimators': 20,
        #    'criterion': 'entropy'
        # },
        # {
        #    'n_estimators': 20,
        #    'criterion': 'gini'
        # },
        # {
        #    'n_estimators': 30,
        #    'criterion': 'gini'
        # },
    ],
    'SVM': [
       # {
       #     'kernel': 'rbf',
       # },
       # {
       #     'kernel': 'linear',
       # },
       # {
       #     'kernel': 'poly',
       # },
       # {
       #     'kernel': 'sigmoid',
       # },
       # {
       #     'kernel': 'precomputed',
       # },
    ],
    #'AdaBoost':[
    #    {
    #        'n_estimators': 50,
    #        'learning_rate': 1.0
    #    }
    #        {
    #            'n_estimators': 1,
    #            'learning_rate': 1.0
    #        },
    #        {
    #            'n_estimators': 500,
    #            'learning_rate': 1.0
    #        },
    #        {
    #            'n_estimators': 50,
    #            'learning_rate': 10.0
    #        },
    #]
    'NeuroNet':[
        # {
        #    'hidden_layer_sizes': (100,),
        #    'solver': 'adam',
        #    'learning_rate_init': 0.001
        # },
        # {
        #     'hidden_layer_sizes': (1,),
        #     'solver': 'adam',
        #     'learning_rate_init': 0.001
        # },
        # {
        #     'hidden_layer_sizes': (100,),
        #     'solver': 'adam',
        #     'learning_rate_init': 0.0001
        # },
        # {
        #     'hidden_layer_sizes': (1000,),
        #     'solver': 'adam',
        #     'learning_rate_init': 0.001
        # },
    ]

}
for classifier in settings.keys():
    print('************************************' + classifier + '************************************')
    for setting in settings[classifier]:
        print('********NewSetting********')
        print('Settings: ' + str(setting))
        overallAccuracy = 0
        timeStart = time.time()



        for i in range(4):              #4 folds

            #trainData = [FoldData() for j in range (3)]

            #3 training folds


            t_features = featureScale(folds[i].features)
            t_labels = folds[i].labels.ravel()

            '''trainData[0].feature_matrix_train = featureScale(folds[i].features)
            trainData[0].labels_train= folds[i].labels.ravel()
            trainData[1].feature_matrix_train = featureScale(folds[(i+1)%4].features)
            trainData[1].labels_train = folds[(i+1)%4].labels.ravel()
            trainData[2].feature_matrix_train = featureScale(folds[(i+2)%4].features)
            trainData[2].labels_train = folds[(i+2)%4].labels.ravel()'''

            #featureMatrixTrain = np.append(trainData[0].feature_matrix_train[0], trainData[1].feature_matrix_train[0], axis = 0)
            #featureMatrixTrain = np.append(featureMatrixTrain, trainData[2].feature_matrix_train[0], axis = 0)
            #labelsTrain = np.append(trainData[0].labels_train, trainData[1].labels_train, axis=0)
            #labelsTrain = np.append(labelsTrain, trainData[2].labels_train, axis=0)

            #1 evaluation fold

            e_features = eval_folds[i].features
            e_labels = eval_folds[i].labels

            featureMatrixTest = featureScale(e_features)
            labelsTest = e_labels.ravel()
            fileNamesTest = eval_folds[i].file_names.ravel()

            #featureClassCoerr(featureMatrixTrain, labelsTrain, range(0,60))

            # training
            model, meanCrossVal = trainModel(t_features[0], t_labels, classifier, setting)

            # testing
            accuracy, precision, recall, f1 = testModel(model, featureMatrixTest[0], labelsTest)
            overallAccuracy += accuracy

            # apply 10 sec. prediction strategy
            evaldict = testModel_per_file(model, featureMatrixTest[0], labelsTest, fileNamesTest)

            # presetting
            classes = {'beach':0, 'bus':0, 'cafe/restaurant':0, 'car':0, 'city_center':0, 'forest_path':0, 'grocery_store':0, 'home':0, 'library':0, 'metro_station':0, 'office':0, 'park':0, 'residential_area':0, 'train':0, 'tram':0}
            longestinterval = 0
            currentinterval = 0
            intervalclass = ''
            previous = ''
            acctrue_FR = 0
            accfalse_FR = 0
            acctrue_IN = 0
            accfalse_IN = 0

            # loop over files/scenes
            for file, scene in evaldict.iteritems():
                for p in scene['predicted']:
                    # strategie frequency
                    classes[p] += 1

                    # strategie interval
                    if (p == previous):
                        currentinterval += 1
                    else:
                        if (longestinterval < currentinterval):
                            intervalclass = p
                            currentinterval = 0
                    previous = p

                #strategie frequency
                frequclass = max(classes.iteritems(), key=operator.itemgetter(1))[0]
                if (frequclass == scene['actual'][0]):
                    acctrue_FR += 1
                else:
                    accfalse_FR += 1
                #reset classes count
                classes = dict.fromkeys(classes, 0)

                # strategie interval
                if (intervalclass == scene['actual'][0]):
                    acctrue_IN += 1
                else:
                    accfalse_IN += 1

            print "Scene accuracy FREQUENCY: #true: " + str(acctrue_FR) + ' #false: ' + str(accfalse_FR) + ' acc: ' + str(acctrue_FR/float(acctrue_FR+accfalse_FR))
            print "Scene accuracy INTERVAL: #true: " + str(acctrue_IN) + ' #false: ' + str(accfalse_IN) + ' acc: ' + str(acctrue_IN / float(acctrue_IN + accfalse_IN))

        print(overallAccuracy/4)

        print("Train/Prediction Time (sec.)",(time.time()-timeStart))