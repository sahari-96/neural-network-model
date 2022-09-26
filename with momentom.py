# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 23:12:10 2021

@author: Asus
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.cross_validation import train_test_split

X, Y = make_moons(n_samples=1000, random_state=42, noise=0.1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#X_train = np.array([[0, 0], [0, 1],
#                [1, 0], [1, 1]])
#Y_train = np.array([0, 1, 1, 0])

print('1: online')
print('2: mini_batch')
print('3: Batch')

train_model = int(input('select a train_model:\n\n'))

print('\n\n1:logistic sigmoid')
print('2: tanh')


opt1 = int(input('select an activation function for hidden layer:\n\n'))


   
print('\n\n1: logistic sigmoid')
print('2: linear')
print('3: tanh')
opt2 = int(input('select an activation function for output layer:\n\n'))

def act1(z):
    if opt1 == 1:
        return 1 / (1 + np.exp(-z))
    elif opt1 == 2:
        return z
    else:
        return np.tanh(z)
    
def act2(z):
    if opt2 == 1:
        return 1 / (1 + np.exp(-z))
    elif opt2 == 2:
        return z
    else:
        return np.tanh(z)
    
    
def dact1_dz(z):
    if opt1 == 1:
        return 1 / (1 + np.exp(-z)) * (1 - (1 / (1 + np.exp(-z))))
    elif opt1 == 2:
        return 1
    else: 
        return 1.0 - z**2 
    
def dact2_dz(z):
    if opt2 == 1:
        return 1 / (1 + np.exp(-z)) * (1 - (1 / (1 + np.exp(-z))))
    elif opt2 == 2:
        return 1
    else: 
        return 1.0 - z**2 




np.set_printoptions(formatter={"float": "{: 0.3f}".format}, linewidth=np.inf)
np.random.seed(1)



n_hidden_nodes = int(input('please input number of hidden layer neurons:\n\n' ))
n_output_nodes = int(input('please input number of output layer neurons:\n\n' ))



n_input_nodes = X_train.shape[1]

W1 = np.random.normal(size=(n_hidden_nodes, n_input_nodes))  
W2 = np.random.normal(size=(n_output_nodes, n_hidden_nodes))  

B1 = np.random.random(size=(n_hidden_nodes, 1))  
B2 = np.random.random(size=(n_output_nodes, 1))  

learning_rate = float(input('please input amount of learningrate:\n\n' ))

print('\n\n1: fixed type')
print('2: redusing type')

l = int(input('select the type of learning rate:\n\n' ))
iterations = int(input('please input number of iterations:\n\n' ))


  

def forward(x, predict=True):
    a0 = x.T
    z1 = W1.dot(a0) + B1
    a1 = act1(z1)
    z2 = W2.dot(a1) + B2
    a2 = act2(z2)
    if predict is False:
        return a0, z1, a1, z2, a2
    return a2

# training network by back propagation
def train(x, y, iterations, learning_rate):
    global W1, W2, B1, B2, error
    m = x.shape[0]
    error = []  
    last_dw1 = np.zeros((n_hidden_nodes, n_input_nodes))
    last_dw2 = np.zeros((n_output_nodes, n_hidden_nodes))
    last_db1 = np.zeros((n_hidden_nodes, 1))
    last_db2 = np.zeros((n_output_nodes, 1))
   
    for _ in range(iterations):
        if train_model == 1:
            for i in range(X_train.shape[0]):
                x = np.array(X_train[i])[np.newaxis];  y = Y_train[i]
               
                a0, z1, a1, z2, a2 = forward(x, predict=False)

            e = a2 - y.T
            dz2 = e * dact2_dz(z2)
            dw2 = (dz2.dot(a1.T) / m) + ((0.9)*(last_dw2))
            db2 = np.sum(dz2, axis=1, keepdims=True) / m + ((0.9)*(last_db2))

            da1 = W2.T.dot(dz2)
            dz1 = np.multiply(da1, dact1_dz(z1))
            dw1 = (dz1.dot(a0.T) / m) + ((0.9)*(last_dw1))
            db1 = (np.sum(dz1, axis=1, keepdims=True) / m) + ((0.9)*(last_db1))
            last_dw1 = dw1
            last_dw2 = dw2
            last_db1 = db1
            last_db2 = db2
        
            if(l == 2):
                learning_rate = learning_rate - ((0.01)*(learning_rate))
            

                W1-= learning_rate * dw1
                B1-= learning_rate * db1
                W2-= learning_rate * dw2
                B2-= learning_rate * db2
        elif train_model == 2:
            
            no_of_batches = len(X_train) // 50
            
            for j in range(no_of_batches):
                 x = X_train[j*50:(j+1)*50]
                 y = Y_train[j*50:(j+1)*50]
                
                    
                 a0, z1, a1, z2, a2 = forward(x, predict=False)

                 e = a2 - y.T
                 dz2 = e * dact2_dz(z2)
                 dw2 = (dz2.dot(a1.T) / m) + ((0.9)*(last_dw2))
                 db2 = np.sum(dz2, axis=1, keepdims=True) / m + ((0.9)*(last_db2))

                 da1 = W2.T.dot(dz2)
                 dz1 = np.multiply(da1, dact1_dz(z1))
                 dw1 = (dz1.dot(a0.T) / m) + ((0.9)*(last_dw1))
                 db1 = (np.sum(dz1, axis=1, keepdims=True) / m) + ((0.9)*(last_db1))
                 last_dw1 = dw1
                 last_dw2 = dw2
                 last_db1 = db1
                 last_db2 = db2
        
                 if(l == 2):
                    learning_rate = learning_rate - ((0.01)*(learning_rate))
            

                 W1-= learning_rate * dw1
                 B1-= learning_rate * db1
                 W2-= learning_rate * dw2
                 B2-= learning_rate * db2
        elif train_model == 3:
            a0, z1, a1, z2, a2 = forward(x, predict=False)

            e = a2 - y.T
            dz2 = e * dact2_dz(z2)
            dw2 = (dz2.dot(a1.T) / m) + ((0.9)*(last_dw2))
            db2 = np.sum(dz2, axis=1, keepdims=True) / m + ((0.9)*(last_db2))

            da1 = W2.T.dot(dz2)
            dz1 = np.multiply(da1, dact1_dz(z1))
            dw1 = (dz1.dot(a0.T) / m) + ((0.9)*(last_dw1))
            db1 = (np.sum(dz1, axis=1, keepdims=True) / m) + ((0.9)*(last_db1))
            last_dw1 = dw1
            last_dw2 = dw2
            last_db1 = db1
            last_db2 = db2
        
            if(l == 2):
                learning_rate = learning_rate - ((0.01)*(learning_rate))


            W1-= learning_rate * dw1
            B1-= learning_rate * db1
            W2-= learning_rate * dw2
            B2-= learning_rate * db2
            
            
        error.append(np.average(e ** 2))
       

    return(error)
    return(W1)
    return(W2)
    return(B1)
    return(B2)
    
    
error = train(X_train, Y_train, iterations, learning_rate)
plt.plot(error)
plt.xlabel("training iterations")
plt.ylabel("mse")

print("W_hidden:", W1)
print("W_output:", W2)