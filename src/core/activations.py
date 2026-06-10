import numpy as np

def relu(X):
  return np.maximum(0,X)

def relu_deriv(X):
   dev= (X>0).astype(int)
   return dev

def sigmoid(X):
  return (1/(1+np.exp(-X)))

def softmax(X):
    X = X - np.max(X, axis=0, keepdims=True)
    exp = np.exp(X)
    return exp / np.sum(exp, axis=0, keepdims=True)
