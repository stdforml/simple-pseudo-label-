# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:53:18 2020

@author: Administrator
"""

#加载包
import gzip
import pickle
import numpy as np

#加载数据函数
def load_data():
    f = gzip.open('F:/pseudol label codes/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='bytes')
    f.close()
    return training_data, validation_data, test_data

def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data



#选择一小部分训练数据函数
def split_by_label(dataset, num_per_label):
    # pick out the same size label from data set
    counter = np.zeros(10)  # for 10 classes
    new_dataset = []
    for i in dataset:
        x, y = i
        if type(y) == np.ndarray:
            y = np.argmax(y)
        if y == 0 and counter[0] < num_per_label:
            new_dataset.append(i)
            counter[0] += 1
            continue
        if y == 1 and counter[1] < num_per_label:
            new_dataset.append(i)
            counter[1] += 1
            continue
        if y == 2 and counter[2] < num_per_label:
            new_dataset.append(i)
            counter[2] += 1
            continue
        if y == 3 and counter[3] < num_per_label:
            new_dataset.append(i)
            counter[3] += 1
            continue
        if y == 4 and counter[4] < num_per_label:
            new_dataset.append(i)
            counter[4] += 1
            continue
        if y == 5 and counter[5] < num_per_label:
            new_dataset.append(i)
            counter[5] += 1
            continue
        if y == 6 and counter[6] < num_per_label:
            new_dataset.append(i)
            counter[6] += 1
            continue
        if y == 7 and counter[7] < num_per_label:
            new_dataset.append(i)
            counter[7] += 1
            continue
        if y == 8 and counter[8] < num_per_label:
            new_dataset.append(i)
            counter[8] += 1
            continue
        if y == 9 and counter[9] < num_per_label:
            new_dataset.append(i)
            counter[9] += 1
            continue
    np.random.shuffle(new_dataset)
    print(counter)
    return new_dataset



def vectorized_result(j):
    e = np.zeros((10,1))
    e[j] = 1.0
    return e

