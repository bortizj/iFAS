#!/usr/bin/env python2.7
import numpy as np
from scipy import optimize
from scipy import special
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import neighbors


def linear(a, x, y):
    return (a[0] + a[1] * x) - y


def quadratic(a, x, y):
    return (a[0] + a[1] * x + a[2] * np.power(x,2)) - y


def cubic(a, x, y):
    return (a[0] + a[1] * x + a[2] * np.power(x, 2) + a[3] * np.power(x, 3)) - y


def exponential(a, x, y):
    return (a[0] + np.exp(a[1] * x + a[2])) - y


def logistic(a, x, y):
    return (a[0] / (1 + np.exp(-(a[1] * x + a[2])))) - y


def complementaryError(a, x, y):
    return (1 - 0.5*(1 + special.erf((x - a[0])/(np.sqrt(2)*a[1])))) - y


def gen_data(x, a, fun_type='linear', noise=0, n_outliers=0, random_state=0):
    if fun_type == 'linear':
        y = a[0] + a[1] * x
    elif fun_type == 'quadratic':
        y = a[0] + a[1] * x + a[2] * np.power(x,2)
    elif fun_type == 'cubic':
        y = a[0] + a[1] * x + a[2] * np.power(x, 2) + a[3] * np.power(x, 3)
    elif fun_type == 'exponential':
        y = a[0] + np.exp(a[1] * x + a[2])
    elif fun_type == 'logistic':
        y = a[0] / (1 + np.exp(-(a[1] * x + a[2])))
    elif fun_type == 'complementaryError':
        y = 1 - 0.5*(1 + special.erf((x - a[0])/(np.sqrt(2)*a[1])))
    else:
        y = a[0] + a[1] * x
        print "Error: Unknown function type. Using linear function instead"
    rnd = np.random.RandomState(random_state)
    error = noise * rnd.randn(x.size)
    outliers = rnd.randint(0, x.size, n_outliers)
    error[outliers] *= 10
    return y + error


def fun_text(fun_type='linear'):
    if fun_type == 'linear':
        a = 'a0 + a1 * x'
    elif fun_type == 'quadratic':
        a = 'a0 + a1 * x + a2 * x**2'
    elif fun_type == 'cubic':
        a = 'a0 + a1 * x + a2 * x**2 + a3 * x**3'
    elif fun_type == 'exponential':
        a = 'a0 + exp(a1 * x + a2)'
    elif fun_type == 'logistic':
        a = 'a0 / (1 + exp(-(a1 * x + a2)))'
    elif fun_type == 'complementaryError':
        a = '1 - 0.5*(1 + erf((x - a0)/(sqrt(2)*a1)))'
    else:
        a = 'a0 + a1 * x'
        print "Error: Unknown function type. Using linear function instead"
    return a


def initial_gues(fun_type='linear'):
    if fun_type == 'linear':
        a = [1., 1.]
    elif fun_type == 'quadratic':
        a = [1., 1., 1.]
    elif fun_type == 'cubic':
        a = [1., 1., 1., 1.]
    elif fun_type == 'exponential':
        a = [1., 1., 1.]
    elif fun_type == 'logistic':
        a = [1., 1., 1.]
    elif fun_type == 'complementaryError':
        a = [10., 1.]
    else:
        a = [1., 1.]
        print "Error: Unknown function type. Using linear function instead"
    return a


def optimize_function(x, y, fun_type='linear'):
    fun = globals()[fun_type]
    res_soft_l1 = optimize.least_squares(fun, initial_gues(fun_type), loss='soft_l1', f_scale=0.1, args=(x, y))
    return res_soft_l1.x


def KnnRegression(X, y, nRep=100):
    pass
    # KnnR_scaler = preprocessing.StandardScaler()
    # nSamples, nfeatures = X.shape
    # model = neighbors.KNeighborsRegressor(n_neighbors=5)
    # PCC = []
    # SROCC = []
    # CCD = []
    # # Results are given after averaging 100 repetitions
    # for ii in range(nRep):
    #     # Selecting randomly the training and testing samples for the SVR model 50 % train 50 % test
    #     train, test = model_selection.train_test_split(range(nSamples), test_size=0.5, random_state=ii)
    #     XTrain = X[train, :]
    #     yTrain = y[train]
    #     XTest = X[test, :]
    #     yTest = y[test]
    #     # Normalizing variables (zero mean and std 1)
    #     KnnR_scaler.fit(XTrain)
    #     # Training and testing models
    #     trainedModel = model.fit(KnnR_scaler.transform(XTrain), yTrain)
    #     yEst = trainedModel.predict(KnnR_scaler.transform(XTest))
    #     # Computing testing performance
    #     PCC.append(stats.pearsonr(yEst, yTest)[0])
    #     SROCC.append(stats.spearmanr(yEst, yTest)[0])
    #     CCD.append(dis_correlation(yEst, yTest))
    # yEst = trainedModel.predict(KnnR_scaler.transform(X))
    # pcc = [np.mean(PCC), np.std(PCC)]
    # srocc = [np.mean(SROCC), np.std(SROCC)]
    # ccd = [np.mean(CCD), np.std(CCD)]
    # print "PCC " + np.str(pcc)
    # print "SROCC " + np.str(srocc)
    # print "CCD " + np.str(ccd)
    # print "Average correlations\n\n\n"
    # print "Correlations with KNNR model"
    # print "PCC " + str(stats.pearsonr(yEst, y)[0])
    # print "ROCC " + str(stats.spearmanr(yEst, y)[0])
    # print "CCD " + str(dis_correlation(yEst, y))
    # return yEst
