import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
from numpy import cov
from numpy.linalg import eig
#========================================================
def eigen_vectors(cov_mat):
    D, W = eig(cov_mat)
    print("cov mat",cov_mat)
    print("eigen vectors", W)
    print("eigen values", D)
    return D,W
print("=================================================")
X=np.asarray([[4,1,3],[2,4,3],[2, 1, 4]])
#cov_mat=np.asarray([[17,-4,4],[-4,1,0],[4,0,17]])
#cov_mat=np.asarray([[6,2],[2,3]])
cov_mat=cov(X)
print(cov_mat)
D,W=eigen_vectors(cov_mat)
D_mat=np.zeros((D.shape[0],D.shape[0])) # convert eigen values to eigen value matrix
for i in range(D.shape[0]):
      D_mat[i,i]=D[i]
print(D_mat)
W_D_WT=np.dot(np.dot(W,D_mat),W.T)
W_D_WT=np.round(W_D_WT,2).astype(int)
print(W_D_WT)

#======================================================
def sort_eigen_vectors(D,W):
    idxs=np.argsort(D)
    W_new=[]
    for i in range(len(idxs),0,-1):
        W_new.append(W[idxs[i-1]]) # get maximum eigen vector and put in W_new
    return D,W
D,W=sort_eigen_vectors(D,W)
#======================================================
# let we keep top k(2) vectors from eigen vectors then
W_k=[]
k=2
for i in range(k):
    W_k.append(W[i])
#=====================================================
W_k=np.asarray(W_k)
Z=np.dot(W_k,X)
print(Z.T)



