import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
#===============================================
def read_data():
    df = pd.read_csv('D:/ServerData/courses/MachineLearning/datasets/flower.csv')
    print(df.columns)
    data=np.asarray(df)
    #classes=np.unique(data[:,4])
    iris_setosa=np.asarray([row for row in data if 'Iris-setosa' in row])
    iris_versicolor=np.asarray([row for row in data if 'Iris-versicolor' in row])
    iris_virginica=np.asarray([row for row in data if 'Iris-virginica' in row])
    iris_setosa_train=iris_setosa[0:30,:]
    iris_setosa_test=iris_setosa[30:,:]
    iris_versicolor_train = iris_versicolor[0:30, :]
    iris_versicolor_test = iris_versicolor[30:, :]
    iris_virginica_train = iris_virginica[0:30, :]
    iris_virginica_test = iris_virginica[30:, :]
    trainset=np.asarray(np.concatenate([iris_setosa_train,iris_versicolor_train,iris_virginica_train],axis=0))
    testset=np.asarray(np.concatenate([iris_setosa_test,iris_versicolor_test,iris_virginica_test],axis=0))
    X_train=trainset[:,0:4]
    y_train=trainset[:,4]
    X_test=testset[:,0:4]
    y_test=testset[:,4]
    y_train[y_train=='Iris-setosa']=0
    y_train[y_train == 'Iris-versicolor'] = 1
    y_train[y_train == 'Iris-virginica'] = 2
    y_test[y_test == 'Iris-setosa'] = 0
    y_test[y_test == 'Iris-versicolor'] = 1
    y_test[y_test == 'Iris-virginica'] = 2
    return X_train,X_test,y_train,y_test
#===============================================
def standardize_data(x_train,x_test):
    # z = (x - u) / s
    ss=StandardScaler()
    ss.fit(x_train)
    z_train=ss.transform(x_train)
    z_test=ss.transform(x_test)
    print(z_train)
    print(z_test)
    return z_train,z_test

def covariance_matrix(x_train):
    feat_cols = ['feature' + str(i) for i in range(x_train.shape[1])]
    pca_flower = PCA(n_components=2)
    pca_flower.fit(x_train)
    z_train=pca_flower.transform(x_train)
    cov=pca_flower.get_covariance()
    #principalComponents_flower = pca_flower.fit_transform(x_train)
    principal_flower_Df = pd.DataFrame(data=z_train
                                       , columns=['principal component 1', 'principal component 2'])
    print(principal_flower_Df.tail())
    print('Explained variation per principal component: {}'.format(pca_flower.explained_variance_ratio_))
    print(z_train)




if __name__=='__main__':
    X_train,X_test,y_train,y_test=read_data()
    X_train,X_test=standardize_data(X_train,X_test)
    covariance_matrix(X_train)