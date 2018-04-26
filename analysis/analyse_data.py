'''
Created: 17.04.2018
@author:
description:
parameter:
'''

import numpy as np
import sklearn as sk
import scipy.stats as stat
import json
import os
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt

#Merge each scene into one big data matrix =>
def merge_data():
    data = read_data_files('/Users/Florian/Documents/DCASE2017_development_set/mfcc_numpy')
    whole_data = None
    i = 0
    cols = len(data.keys()*501)
    matrix = np.empty([60, cols])
    #print(matrix.shape)
    for file_ID in data:
        col_offset = i*501
        #print(file_ID)
        for row in range(len(data[file_ID])):
            #print(len(data[file_ID][row][500]))
            matrix[row][col_offset:((i+1)*501)] = data[file_ID][row]
        i = i + 1

    return matrix


def read_data_files(path):
    meta_data = np.loadtxt('../iodata/data/meta.txt', dtype={'names': ('fileID', 'label', 'labelID'),
                                         'formats': ('S100', 'S100', 'S100')}, delimiter='\t')
    print(meta_data)
    data = {}
    whole_json_file = path + '/whole.npy'

    if (os.path.exists(whole_json_file)):
        return np.load(whole_json_file).item()

    for item in meta_data:
        file_ID = item['fileID'].replace('audio/', '').replace('.wav', '')
        file_path = path + '/' + file_ID + '.npy'
        print('Process file:' + file_path)
        data[file_ID] = np.load(file_path)

    np.save(whole_json_file, data)

    return data


def print_correlation_matrix_merged_data(merged_matrix):

    corr_matrix = np.corrcoef(merged_matrix)
    fig, ax = plt.subplots()
    cax = ax.imshow(corr_matrix, cmap='hot', interpolation='nearest')
    fig.colorbar(cax)
    fig.savefig('heatmap')
    plt.subplots_adjust(left=0.3, right=0.9, bottom=0.3, top=0.9)
    ax.set_aspect("equal")
    #plt.show()
    np.savetxt('Correlation_Table.csv', corr_matrix, fmt='%.2f', delimiter=';')


def print_Description(merged_matrix):
    file_name = 'Descriptions.csv'

    with open(file_name, 'w') as f:
        f.write('Feature; nobs; min; max; mean; variance; skeweness; kurtosis; median; Q1; median; Q3')
        f.write('\n')

    for row in range(0, len(merged_matrix)):
        print('Feature: ' + str((row + 1)))
        with open(file_name, 'a') as f:
            line = 'Feature ' + str(row) + '; ' + \
                   str(stat.describe(merged_matrix[row])).replace('(', '').replace(')', '').replace(',',';') + \
                   ';' + str(stat.nanmedian(merged_matrix[row]))

            quantile_arr=list(stat.mstats.mquantiles(merged_matrix[row]))

            print(str(quantile_arr))

            line = line + '; '.join([str(quantile) for quantile in quantile_arr])
            line = line.replace('.', ',')
            f.write(line)
            f.write('\n')


def print_boxplots(merged_matrix):
    for row in range(0, len(merged_matrix)):
        print('Feature: ' + str((row + 1)))
        fig, ax = plt.subplots()
        bp = plt.boxplot(merged_matrix[row])
        print(bp['medians'][0])
        print(bp['fliers'])
        print(bp['medians'][0].get_ydata())
        plt.xlabel('Feature : ' +  str(row+1))
        plt.tight_layout()
        fig.savefig('img/Feature_' + str(row+1))
        plt.close()


#for row in range(0, len(merged_matrix) - 1):
    #print(str(row) + ':')
    #print(pearsonr(merged_matrix[row], merged_matrix[row+1]))
    #print(np.corrcoef(merged_matrix[row], merged_matrix[row+1]))
#readData('/Users/Florian/Documents/DCASE2017_development_set/mfcc_numpy')

merged_matrix = merge_data()
#print_correlation_matrix_merged_data(merged_matrix)
print_Description(merged_matrix)
#print_boxplots(merged_matrix)