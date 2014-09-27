# Python 2.7 users.
from __future__ import print_function
from __future__ import division

#Testing on the Iris data set
from sklearn.datasets import load_iris
from sklearn.cross_validation import KFold

#Classifiers to compare
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#Score metric
from sklearn.metrics import precision_score
#Bayes optimization
from bayes_opt.bo.bayes_opt import bayes_opt


# Load data set and target values
data = load_iris()['data']
target = load_iris()['target']

# LogisticRegression CV
def LR_CV(c, ints, kf):

    clf = LogisticRegression(C = c, intercept_scaling = ints, random_state = 2)
    res = 0

    for itr, ite in kf:
        xtrain, xtest = data[itr], data[ite]
        ytrain, ytest = target[itr], target[ite]

        clf.fit(xtrain, ytrain)
        pred = clf.predict(xtest)

        res += precision_score(ytest, pred)

    return res/5

# SVM CV
def SVR_CV(c, g, e):

    clf = SVC(C = c, gamma = g, tol = e, random_state = 2)
    res = 0

    for itr, ite in kf:
        xtrain, xtest = data[itr], data[ite]
        ytrain, ytest = target[itr], target[ite]

        clf.fit(xtrain, ytrain)
        pred = clf.predict(xtest)

        res += precision_score(ytest, pred)

    return res/5

#Random Forest CV
def RF_CV(trees, split, leaf):
    
    clf = RandomForestClassifier(\
        n_estimators = int(trees),\
        min_samples_split = int(split),\
        min_samples_leaf = int(leaf),\
        max_features = None,\
        random_state = 3,\
        n_jobs = -1)

    res = 0
    
    for itr, ite in kf:
        xtrain, xtest = data[itr], data[ite]
        ytrain, ytest = target[itr], target[ite]

        clf.fit(xtrain, ytrain)
        pred = clf.predict(xtest)

        res += precision_score(ytest, pred)

    return res/5


if __name__ == "__main__":

    kf = KFold(len(target), n_folds = 5, shuffle = True, random_state = 1)

    # Search for a good set of parameters for logistic regression
    bo_LR = bayes_opt(lambda c, ints: LR_CV(c, ints, kf), {'c' : (0.001, 100), 'ints' : (0.001, 100)})
    ylr, xlr = bo_LR.log_maximize(num_it = 25)

    # Search for a good set of parameters for support vector machine
    bo_SVR = bayes_opt(SVR_CV, {'c' : (0.001, 100), 'g' : (0.0001, 1), 'e' : (0.001, 10)})
    bo_SVR.initialize({'c' : 0.1, 'g' : .01, 'e' : 0.005})
    ysvr, xsvr = bo_SVR.log_maximize(init_points = 5, restarts = 15, num_it = 25)

    # Search for a good set of parameters for random forest.
    bo_RF = bayes_opt(RF_CV, {'trees' : (10, 200), 'split' : (2, 20), 'leaf' : (1, 10)})
    yrf, xrf = bo_RF.maximize(init_points = 5, restarts = 15, num_it = 25)
