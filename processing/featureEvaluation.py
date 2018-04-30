'''
Created: 30.04.2018
@author: davidgraf
description:
parameter:
'''

from sklearn.feature_selection import mutual_info_classif

def featureClassCoerr(featureMatix, labelMatrix, features):

    mInfo = mutual_info_classif(featureMatix, labelMatrix)
    print "------------------------- feature evaulation mutual_info_classif 2 begin -------------------------"

    for i in range(len(mInfo)):
        print str(features[i]) + ": " + str(mInfo[i])

    print "------------------------- feature evaulation mutual_info_classif 2 end -------------------------"
