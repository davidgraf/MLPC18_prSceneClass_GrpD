'''
Created: 17.04.2018
@author:
description:
parameter:
'''

import os
import numpy as np
import pandas as pd
import glob

DATA_DIR = ""

def writeToCSV (matrix, filename):

    myfile = open(DATA_DIR + filename, "w")
    for l in matrix:
        try:
            if isinstance(l, (list,)) or isinstance(l, (np.ndarray,)):
                for v in l:
                    myfile.write(str(v) + "\t")
            else:
                myfile.write(str(l))
            myfile.write("\n")
        except:
            myfile.write("Error in writing file "+filename)

    myfile.close()

    print "data exported to " + filename

