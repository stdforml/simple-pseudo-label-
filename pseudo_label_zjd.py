# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 10:22:16 2020

@author: Administrator
"""

import numpy as np
import sys
sys.path.append(r'F:/pseudol label codes/')
import data_load_zjd 



#functions load
class Pseudo_label(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.default_weight_initializer()
        
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def cost_derivative(self, output_activations, y):
            return (output_activations-y)
    
    
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]  
        # feedforward
        activation = x
        activations = [x]  
        zs = []  
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)  
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])  
        nabla_b[-1] = delta  
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())  
    
        for l in range(2, self.num_layers):
            z = zs[-l]  
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return (nabla_b, nabla_w)  
    

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]  
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  
        self.weights = [w - (eta / len(mini_batch)) * nw  
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]

    

    def feedforward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    

    def pred_label(self, unlabeled_data):
        pred_labels = [np.argmax(self.feedforward(x))  for x in unlabeled_data]
        pred_result = [data_load_zjd.vectorized_result(y) for y in pred_labels]
        return pred_result
    

    def evaluate(self, labeled_data):
            labeled_results = [(np.argmax(self.feedforward(x)), y)
                            for (x, y) in labeled_data]
            return sum(int(x == np.argmax(y)) for (x, y) in labeled_results)
    

    def evaluate_t(self, unlabeled_data):
            unlabeled_results = [(np.argmax(self.feedforward(x)), y)
                            for (x, y) in unlabeled_data]
            return sum(int(x == y) for (x, y) in unlabeled_results)
        
        
        
    def train_generator(self, labeled_data, unlabeled_data, mini_batch_size_LB, mini_batch_size_ULB):
        mini_batches = []
        flag_joins = []
        n_labeled_data = len(labeled_data)
        n_unlabeled_data = len(unlabeled_data)
        
        np.random.shuffle(labeled_data)
        np.random.shuffle(unlabeled_data)
        mini_batches_lb = [labeled_data[k:k+mini_batch_size_LB]
                           for k in range(0, n_labeled_data, mini_batch_size_LB)]
        mini_batches_ulb = [unlabeled_data[k:k+mini_batch_size_ULB]
                            for k in range(0, n_unlabeled_data, mini_batch_size_ULB)]
        for i in range(len(mini_batches_lb)):
            flag_join = np.r_[np.repeat(0.0, mini_batch_size_LB), np.repeat(1.0, mini_batch_size_ULB)].reshape(-1,1)
            mini_batch_com = np.vstack((mini_batches_lb[i], mini_batches_ulb[i]))
            indices = np.arange(flag_join.shape[0])
            np.random.shuffle(indices)
            mini_batches.append(mini_batch_com[indices])
            flag_joins.append(flag_join[indices])
        return mini_batches, flag_joins
            
         

    def update_mini_batch_pl(self, mini_batch, eta, alpha_t):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]  
        for x, y, flag in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            if flag < 1.0:
                nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  
                nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            else:
                nabla_b = [nb + alpha_t*dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  
                nabla_w = [nw + alpha_t*dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  
        self.weights = [w - (eta / len(mini_batch)) * nw  
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]
        
        
        
    def PL_SGD(self, labeled_data, unlabeled_data, epochs, 
               mini_batch_size_LB, mini_batch_size_ULB, eta, test_data=None):
        n = len(labeled_data)
        n_vali = len(unlabeled_data)
        if test_data: n_test = len(test_data)
        unlabeled_data_orig = [x for x, y in unlabeled_data]
        
        for j in range(epochs):
            if j < 10:
                np.random.shuffle(labeled_data)
                mini_batches = [labeled_data[k:k + mini_batch_size_LB]
                                for k in range(0, n, mini_batch_size_LB)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta)
                pseudo_labels = self.pred_label(unlabeled_data_orig)
                unlabeled_data_update = list(zip(unlabeled_data_orig, pseudo_labels))     #no one-hot data
            elif 10 <= j < 60:
                alpha_t = (j-10)/50.0 * 3
                mini_batches, flag_joins = self.train_generator(labeled_data, unlabeled_data_update,
                                                    mini_batch_size_LB, mini_batch_size_ULB)
                mini_batches = np.c_[mini_batches, flag_joins]
                for mini_batch in mini_batches:
                    self.update_mini_batch_pl(mini_batch, eta, alpha_t)
                pseudo_labels = self.pred_label(unlabeled_data_orig)
                unlabeled_data_update = list(zip(unlabeled_data_orig, pseudo_labels))
            elif 60 <= j:
                alpha_t = 3
                mini_batches, flag_joins = self.train_generator(labeled_data, unlabeled_data_update,
                                                                mini_batch_size_LB, mini_batch_size_ULB)
                mini_batches = np.c_[mini_batches, flag_joins]
                for mini_batch in mini_batches:
                    self.update_mini_batch_pl(mini_batch, eta, alpha_t)
                pseudo_labels = self.pred_label(unlabeled_data_orig)
                unlabeled_data_update = list(zip(unlabeled_data_orig, pseudo_labels))
            if test_data:
                print("Epoch {0}: labeled:{1}/{2}".format(j, self.evaluate(labeled_data), n))
                print("Epoch {0}: unlabeled:{1}/{2}".format(j, self.evaluate_t(unlabeled_data), n_vali))
                print("Epoch {0}: test:{1}/{2}".format(j, self.evaluate_t(test_data), n_test))
            else:
                print("Epoch {0}: labeled:{1}/{2}".format(j, self.evaluate(labeled_data), n))
                print("Epoch {0}: unlabeled:{1}/{2}".format(j, self.evaluate_t(unlabeled_data), n_vali))
                print("Epoch {0} complete".format(j))
    



def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))





if __name__ == "__main__":
    training_data, validation_data, test_data = data_load_zjd.load_data_wrapper()
    #选在一小部分训练数据中进行训练
    training_data_small = data_load_zjd.split_by_label(training_data, num_per_label=1000)

    net = Pseudo_label([784, 30, 10])
    net.PL_SGD(labeled_data=training_data_small, unlabeled_data=validation_data, 
               epochs=100, mini_batch_size_LB = 100, mini_batch_size_ULB = 100, eta=3.0, test_data=test_data)






#training_data 50000
#test_data 10000








