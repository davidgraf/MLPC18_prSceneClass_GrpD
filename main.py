'''
Created: 17.04.2018
@author: davidgraf
description: main application to run
parameter:
'''

DATA_DIR = "C:/Temp/DCASE2017_development_set"

# imports
from iodata.readData import read_data_files

# read data
data = read_data_files(DATA_DIR)

# data analysis
# ... analysis(data)

# preprocssing (feature scaling, feature evaluation, feature selection)
# ...

# training
model, meanCrossVal = trainModel(featuresTrain, labelsTrain)

# testing
accuracy, precision, recall, f1 = testModel(model, featuresTest, labelsTest)

















