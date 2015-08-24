__author__ = 'mateuszopala'
from sklearn.svm import SVR
import numpy as np
from utils.utils import generate_submission
import pandas as pd
from munging.loaders import SupervisedLoader, TestSetLoader
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.grid_search import GridSearchCV
import cPickle


def grid_search_for_svr():
    train_x, train_y = SupervisedLoader.load('../data')
    gammas = [4.]
    clf = SVR(verbose=1)
    param_grid = {'gamma': gammas, 'C': [10., 20., 30., 40.]}
    grid_search = GridSearchCV(clf, param_grid, scoring='mean_squared_error', n_jobs=4, verbose=1)

    grid_search.fit(train_x, train_y)

    print grid_search.best_score_
    print grid_search.best_params_

    with open('../data/another2_svr.pkl', 'wb') as f:
        cPickle.dump(grid_search.best_estimator_, f)


if __name__ == "__main__":
    grid_search_for_svr()
    # train_x = np.load('../data/features.npy')
    # train_y = np.load('../data/hazards.npy')
    # test_x = np.load('../data/test_x.npy')
    # ids = np.load('../data/test_ids.npy')
    # test_x, ids = TestSetLoader.load('../data')
    # train_x, train_y = SupervisedLoader.load('../data')
    # with open('../data/another_svr.pkl', 'rb') as f:
    #     clf = cPickle.load(f)
    # print 'svr loaded'
    #
    # # clf = SVR(C=100., verbose=True, cache_size=4000, epsilon=1.e-2)
    # #
    # # clf = RandomForestRegressor(n_estimators=10000, n_jobs=4, verbose=True)
    # # clf = KNeighborsRegressor(n_neighbors=3) # worse definitely than SVM and RandomForestRegressor
    #
    #
    # # clf.fit(train_x, train_y)
    #
    # print 'SVR fitted'
    #
    # predicted = clf.predict(test_x)
    #
    # print predicted
    #
    # generate_submission(list(ids.astype(np.uint32)), list(predicted), '../data/submission_best_svr.csv')





