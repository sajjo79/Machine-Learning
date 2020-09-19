from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd

#===================================================
def read_data(read_numpy):
    if read_numpy==True:
        X_train_1=np.load('X_train.npz',allow_pickle=True)
        X_test_1 = np.load('X_test.npz',allow_pickle=True)
        y_train_1 = np.load('y_train.npz',allow_pickle=True)
        y_test_1 = np.load('y_test.npz',allow_pickle=True)

        X_train=X_train_1['X_train']
        X_test = X_test_1['X_test']
        y_train = y_train_1['y_train']
        y_test = y_test_1['y_test']
    else:
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

        print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
        np.savez_compressed('X_train.npz',X_train=X_train)
        np.savez_compressed('X_test.npz', X_test=X_test)
        np.savez_compressed('y_train.npz', y_train=y_train)
        np.savez_compressed('y_test.npz', y_test=y_test)

    return X_train,X_test,y_train,y_test
#===================================================
def train_model(X_train,y_train):
    y_train=y_train.astype('int')
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn


def print_accuracy(f, X_test,y_test):
    print("Accuracy = {0}%".format(100*np.sum(f(X_test) == y_test)/len(y_test)))

#===================================================
def test_model(model,X_test,y_test):
    # for i in range(len(X)):
    #      val=X[i]
    #      val=np.expand_dims(val,axis=0)
    #      val = np.expand_dims(val, axis=0)
    #      pred = model.predict(val)
    #      print(val,y_test[i],pred)
    preds=model.predict(X_test)
    print("Accuracy = {0}%".format(100 * np.sum(preds == y_test) / len(y_test)))


if __name__=='__main__':
    X_train,X_test,y_train,y_test=read_data(True)
    for i in range(4):
        print('Results on feature ',i)
        X_tr=X_train[:,i]
        X_tr=np.reshape(X_tr,(-1,1))
        X_te=X_test[:,i]
        X_te=np.reshape(X_te,(-1,1))
        model=train_model(X_tr,y_train)
        test_model(model,X_te,y_test)
    print("======================================================")
    for i in range(3): # F1, F2, and F4
        print('Results on feature ',i,3)
        col1,col2=X_train[:,i],X_train[:,3]
        col1=np.expand_dims(col1,axis=1)
        col2 = np.expand_dims(col2, axis=1)
        X_tr=np.concatenate((col1,col2),axis=1)
        model = train_model(X_tr, y_train)

        col1, col2 = X_test[:, i], X_test[:, 3]
        col1 = np.expand_dims(col1, axis=1)
        col2 = np.expand_dims(col2, axis=1)
        X_te = np.concatenate((col1, col2), axis=1)
        test_model(model,X_te,y_test)

    print('Results on all features')
    model = train_model(X_train, y_train)
    test_model(model, X_test, y_test)