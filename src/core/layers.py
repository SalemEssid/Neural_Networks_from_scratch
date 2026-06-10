import numpy as np
from core.activations import relu, softmax, relu_deriv
def linear_layer(W,A,b):
    Z=W.dot(A)+b
    return Z

def forward_propagation(X,parameters):
    W1=parameters["W1"]
    W2=parameters["W2"]
    W3=parameters["W3"]
    b1=parameters["b1"]
    b2=parameters["b2"]
    b3=parameters["b3"]

    Z1=linear_layer(W1,X,b1)
    A1=relu(Z1)

    Z2=linear_layer(W2,A1,b2)
    A2=relu(Z2)

    Z3=linear_layer(W3,A2,b3)
    A3=softmax(Z3)
    cache={"Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2,"Z3":Z3,"A3":A3}
    return cache

def backward_propagation(X,Y,cache,parameters):
    A3=cache["A3"]
    A2=cache["A2"]
    A1=cache["A1"]
    Z2=cache["Z2"]
    Z1=cache["Z1"]
    W1=parameters["W1"]
    W2=parameters["W2"]
    W3=parameters["W3"]
    m=X.shape[1]

    dZ3 = A3 - Y
    dW3 = (1/m) * dZ3.dot(A2.T)
    db3 = (1/m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = W3.T.dot(dZ3)
    dZ2 = dA2 * relu_deriv(Z2)

    dW2 = (1/m) * dZ2.dot(A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * relu_deriv(Z1)

    dW1 = (1/m) * dZ1.dot(X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    grads={"dW1":dW1,"db1":db1,"dW2":dW2,"db2":db2,"dW3":dW3,"db3":db3}
    return grads