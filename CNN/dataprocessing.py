# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 16:53:23 2018

@author: bx_chen
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

CUT_NUM = 1 #去掉第一个样本

def get_figures(data_path, batch_id):
    _pkl_path = open(data_path + '/figure_gather_c' + str(batch_id) + '.pkl', mode='rb+')
    _data_feature = pickle.load(_pkl_path)
    _data = np.array(_data_feature)

    _list = []
    for i in range(CUT_NUM,len(_data)):
        _temp = []
        for j in range(7):
            _temp.append(_data[i][j])
        _temp = np.transpose(np.array(_temp)) #沿Z轴数组转置
        _list.append(_temp)
        
    return np.array(_list)

def get_labels(data_path, batch_id, flute=1):
    _pkl_path = open(data_path + '/figure_gather_c' + str(batch_id) + '_label.pkl', mode='rb+')
    _data_labels = pickle.load(_pkl_path)
    data = np.array(_data_labels)
    return data[CUT_NUM:,flute-1].reshape(314,1)
    
def get_batches_from(features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

#test
if __name__=='__main__':          

    labels = get_figures('data', 1)





   
