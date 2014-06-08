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
from bayes_opt import bayes_opt

# Load data set and target values
data = load_iris()['data']
target = load_iris()['target']

# LogisticRegression CV
def LR_CV(params):
    kf = KFold(len(target), n_folds = 5, shuffle = True, random_state = 1)
    res = 0
    for itr, ite in kf:
        xtrain, xtest = data[itr], data[ite]
        ytrain, ytest = target[itr], target[ite]

        clf = LogisticRegression(C = params[0], intercept_scaling = params[1], random_state = 2)

        clf.fit(xtrain, ytrain)
        pred = clf.predict(xtest)

        res += precision_score(ytest, pred)

    return res/5

# SVM CV
def SVR_CV(params):
    kf = KFold(len(target), n_folds = 5, shuffle = True, random_state = 1)
    res = 0
    for itr, ite in kf:
        xtrain, xtest = data[itr], data[ite]
        ytrain, ytest = target[itr], target[ite]

        clf = SVC(C = params[0], gamma = params[1], tol = params[2], random_state = 2)

        clf.fit(xtrain, ytrain)
        pred = clf.predict(xtest)

        res += precision_score(ytest, pred)

    return res/5

#Random Forest CV
def RF_CV(params):
    kf = KFold(len(target), n_folds = 5, shuffle = True, random_state = 1)
    res = 0
    for itr, ite in kf:
        xtrain, xtest = data[itr], data[ite]
        ytrain, ytest = target[itr], target[ite]

        clf = RandomForestClassifier(n_estimators = int(params[0]),\
                                     min_samples_split = int(params[1]),\
                                     min_samples_leaf = int(params[2]), max_features = None,\
                                     random_state = 3, n_jobs = -1)

        clf.fit(xtrain, ytrain)
        pred = clf.predict(xtest)

        res += precision_score(ytest, pred)

    return res/5


if __name__ == "__main__":

    # Search for a good set of parameters for logistic regression
    bo_LR = bayes_opt(LR_CV, [(0.001, 100), (0.001, 100)])
    ylr, xlr = bo_LR.log_maximize(num_it = 25)

    # Search for a good set of parameters for support vector machine
    #bo_SVR = bayes_opt(SVR_CV, [(0.001, 100), (0.0001, 1), (0.001, 10)])
    #ysvr, xsvr = bo_SVR.log_maximize(init_points = 5, restarts = 15, num_it = 25)

    # Search for a good set of parameters for random forest.
    #bo_RF = bayes_opt(RF_CV, [(10, 200), (2, 20), (1, 10)])
    #yrf, xrf = bo_RF.maximize(init_points = 5, restarts = 15, num_it = 25)
