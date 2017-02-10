from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
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
    df = pd.read_csv('auction.csv')
    X = df.drop(['SalePrice'], axis = 1).values
    y = df['SalePrice'].values
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size = 0.3)


    lr_hat, score, beta = lassoreg(X_train, X_test, y_train, y_test, alp=0.1)
    print score
    print beta
    print pvalue
    print cross_val_score(Lasso(), X, y)
    # lr_hat, score, beta, pvalue= lassoreg(X_train, X_test, y_train, y_test, alp)
    # np.savetxt('median_benchmark.csv', lr_hat)
