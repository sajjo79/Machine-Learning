import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.linalg as la
#=====================================================
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
#=====================================================
"""
Algorithm steps
STEP 1: STANDARDIZATION
STEP 2: COVARIANCE MATRIX COMPUTATION
STEP 3: COMPUTE THE EIGENVECTORS AND EIGENVALUES OF THE COVARIANCE MATRIX
STEP 4: FEATURE VECTOR
STEP 5: RECAST THE DATA ALONG THE PRINCIPAL COMPONENTS AXES
"""
#======================================================
def standardize_data(x_train,x_test):
    # z=(feature_value - mean)/standard_deviation
    f0 = x_train[:, 0]
    f1 = x_train[:, 1]
    f2 = x_train[:, 2]
    f3 = x_train[:, 3]

    f0_test = x_test[:, 0]
    f1_test = x_test[:, 1]
    f2_test = x_test[:, 2]
    f3_test = x_test[:, 3]

    mean_f0 = np.mean(f0)
    mean_f1 = np.mean(f1)
    mean_f2 = np.mean(f2)
    mean_f3 = np.mean(f3)

    std_f0 = np.std(f0)
    std_f1 = np.std(f1)
    std_f2 = np.std(f2)
    std_f3 = np.std(f3)

    # transformed training and test data to z-subspace
    z0_tr = (f0 - mean_f0) / std_f0
    z1_tr = (f1 - mean_f1) / std_f1
    z2_tr = (f2 - mean_f2) / std_f2
    z3_tr = (f3 - mean_f3) / std_f3

    z0_te = (f0_test - mean_f0) / std_f0
    z1_te = (f1_test - mean_f1) / std_f1
    z2_te = (f2_test - mean_f2) / std_f2
    z3_te = (f3_test - mean_f3) / std_f3

    z_train=np.asarray([z0_tr,z1_tr,z2_tr,z3_tr])
    z_test = np.asarray([z0_te, z1_te, z2_te, z3_te])
    z_train=z_train.T
    z_test=z_test.T
    print('z_train::',z_train)
    print('z_test::',z_test)
    """
    z_train
    [[-0.9394866965817561 1.0127652695244411 - 1.3458216392185802   - 1.3316413131814744]
     [-1.176332082274635 - 0.13232109049062973 - 1.3458216392185802 - 1.3316413131814744]
     [-1.413177467967515 0.32571345351539904 - 1.401819765677051    - 1.3316413131814744]
    [-1.5316001608139556  0.09669618151238464 - 1.2898235127601094  - 1.3316413131814744]
    z_test
    [[-1.2947547751210757 0.09669618151238464 -1.2338253863016386  -1.3316413131814744]
 [-0.5842186180424356 0.7837479975214268 -1.2898235127601094   -1.065903578843708]
 [-0.8210640037353156 2.3868689015425253 -1.2898235127601094   -1.4645101803503575]
 [-0.4657959251959961 2.6158861735455408 -1.3458216392185802   -1.3316413131814744]
    """
    return z_train,z_test
#======================================================
def covariance_matrix(x_train):
    f0 = x_train[:, 0]
    f1 = x_train[:, 1]
    f2 = x_train[:, 2]
    f3 = x_train[:, 3]
    means=[]
    means.append(np.mean(f0))
    means.append(np.mean(f1))
    means.append(np.mean(f2))
    means.append(np.mean(f3))
    means=np.asarray(means)

    N=len(f0)   # number of rows in dataset
    cov_mat=np.zeros([4,4])
    for i in range(4):
        for j in range(4):
            vals=(x_train[:, i]-means[i])*(x_train[:, j]-means[j])
            #print(vals,np.sum(vals))
            cov_mat[i, j]=np.sum(vals)/N
    print('Σ=',cov_mat)
    """
    Σ= [[ 1.         -0.15474952  0.87057662  0.79864619]
 [-0.15474952  1.         -0.4748974  -0.41422871]
 [ 0.87057662 -0.4748974   1.          0.96493269]
 [ 0.79864619 -0.41422871  0.96493269  1.        ]]
    """
    return cov_mat
#======================================================
def eigenvalues_eigen_vectors(cov_mat):
    # Ax=𝜆x
    eig_values,eig_vectors = la.eig(cov_mat)
    eig_vectors=eig_vectors.tolist()

    print("============eigen==================")
    for i in range(4):
        print('𝜆=',eig_values[i])
        print('x',eig_vectors[i])
    eig_vectors=np.asarray(eig_vectors)
    return eig_values, eig_vectors

def print_eig_vectors(eig_values,eig_vectors):
    print("=====================================================")
    print('eigen vectors - principal components in descending order ')
    actual_eig_vals = eig_values.copy()
    for j in range(4):
        idx=np.argmax(eig_values)
        idx_val=np.max(eig_values)
        ind=np.where(actual_eig_vals == idx_val)
        print("Feature, Eigen Value, Eigen Vector ==>",ind[0][0],eig_values[idx],eig_vectors[idx])
        eig_values=np.delete(eig_values,idx)
        del eig_vectors[idx]

#======================================================
def project_data(vectors,x_test):
    # project data on the new axes
    # vectors=(), x_test=()
    projected_data = vectors.T.dot(x_test.T) #(4x4)(4x90)->(4x90)
    print("=====================================================")
    print("projected data", projected_data.T) # (90x4)

#======================================================
if __name__=='__main__':
    X_train,X_test,y_train,y_test=read_data()
    X_train,X_test=standardize_data(X_train,X_test)
    cov_mat=covariance_matrix(X_train)
    eig_vals,eig_vecs=eigenvalues_eigen_vectors(cov_mat)
    print('Choose top k features that you want to keep')
    project_data(eig_vecs,X_train)
    # Reconstruction Error etc. 
