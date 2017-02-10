from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zipfile import ZipFile
from sklearn.linear_model import LinearRegression

def dataclean(filename):
    zf = ZipFile(filename)
    df = pd.read_csv(zf.open(filename))
    productgroup = pd.get_dummies(df['ProductGroup'])
    df = pd.concat([df, productgroup], axis = 1)
    state = pd.get_dummies(df['state'])
    df = pd.concat([df, state], axis = 1)
    df=df.loc[df['YearMade']>1000]
    df=df.loc[df['MachineHoursCurrentMeter']>0.0]
    df = df.dropna()

    #create X and y
    y = df['SalePrice']
    X = df[['Northeast','South', 'West', 'SalesID','ProductGroup_BL','ProductGroup_MG','ProductGroup_SSL','ProductGroup_TEX',
     'ProductGroup_TTT','ProductGroup_WL','SalePrice','YearMade', 'MachineHoursCurrentMeter']]
    return X, y


def linereg(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_hat = lr.predict(X_test)
    score = lr.score(X_train, y_train)
    beta = lr.coef_
    F, pvalue = f_regression(X_train, y_train, center=True)
    return lr_hat, score, beta, pvalue

def lassoreg(X_train, X_test, y_train, y_test, alp):
    ls = Lasso(alpha = alp)
    ls.fit(X_train, y_train)
    ls_hat = ls.predict(X_test)
    score = ls.score(X_train, y_train)
    beta = ls.coef_
    return ls_hat, score, beta

def crossvalscore(estimator, X, y):
    score = cross_val_score(estimator, X, y)
    return score


if __name__ == '__main__':
    df = pd.read_csv('newauction.csv')
    X = df.drop(['SalePrice'], axis = 1).values
    y = df['SalePrice'].values
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.3)

    lr_hat, score1, beta1, pvalue = linereg(X_train, X_test, y_train, y_test)
    ls_hat, score2, beta2 = lassoreg(X_train, X_test, y_train, y_test, alp=1)



    print score1, score2
    print beta1, beta2
    # print pvalue
    print cross_val_score(LinearRegression(), X_train, y_train), cross_val_score(Lasso(), X_train, y_train)

    # lr_hat, score, beta, pvalue= lassoreg(X_train, X_test, y_train, y_test, alp)
    # np.savetxt('median_benchmark.csv', lr_hat)
