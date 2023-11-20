import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold 
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt

def run():
    nan = np.nan
    filename = 'files/marketing.csv'

    dataframe = pd.read_csv(filename)
    data = dataframe.values

    ix = [i for i in range(data.shape[1]) if i != 13]
    X, y = data[:, ix], data[:, 13]
    classlabel = dataframe.iloc[:, -1]
    attr = dataframe.iloc[:, 0:-1]

    k=5
    kf = KFold(n_splits=k, random_state=None)
    model = Pipeline([('i', KNNImputer(n_neighbors=5)), ('m', RandomForestClassifier())])
    model.fit(X,y)

    acc_score = []

    for train_index , test_index in kf.split(attr):
        X_train , X_test = attr.iloc[train_index,:],attr.iloc[test_index,:]
        y_train , y_test = classlabel[train_index] , classlabel[test_index]
        
        model.fit(X_train,y_train)
        pred_values = model.predict(X_test)
        
        acc = metrics.accuracy_score(pred_values , y_test)
        acc_score.append(acc)
        
    avg_acc_score = sum(acc_score)/k
    
    print('accuracy of each fold - {}'.format(acc_score))
    print('Avg accuracy : {}'.format(avg_acc_score))
    model.named_steps["i"]