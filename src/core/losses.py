import numpy as np
def computing_cost (A3,Y):
   m = Y.shape[1]

   loss = -np.sum(Y * np.log(A3 + 1e-8)) / m
   return loss