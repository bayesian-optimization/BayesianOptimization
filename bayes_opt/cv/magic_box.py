'''
Black-box, self optimizing, classification and regression tool.

Still under construction!
'''



import numpy
from ..bo.bayes_opt import bayes_opt

from sklearn import metrics
from sklearn.cross_validation import cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


classification_scores = {'accuracy' : metrics.accuracy_score,\
                        'average_precision' : metrics.average_precision_score,\
                        'f1' : metrics.f1_score,\
                        'precision' : metrics.precision_score,\
                        'recall' : metrics.recall_score,\
                        'roc_auc' : metrics.roc_auc_score,}



class magic_box_classifier:

   


    def __init__(self, score_metric, n_folds = 5, n_jobs = 1, iterations = 50):      

        self.sm = score_metric
        self.folds = n_folds
        self.njobs = n_jobs
        self.it = iterations

        self.logit_argmax = {'C' : 1, 'intercept_scaling' : 1}
        self.randf_argmax = {'n_estimators' : 10, 'min_samples_split' : 2}

        self.coefs = {'c_logit' : 0.3333, 'c_svm' : 0.3333, 'c_randf' : 0.3333}

        self.logit_model = 0
        self.randf_model = 0
        self.svm_model = 0


    def logit_cv(self, C, intercept_scaling):
        return numpy.mean(cross_val_score(estimator = LogisticRegression(C = C, intercept_scaling = intercept_scaling),\
                                      X = self.x, y = self.y, scoring = self.sm, cv = self.folds, n_jobs = self.njobs))

    def svm_cv(self, C, gamma):
        return numpy.mean(cross_val_score(estimator = SVC(C = C, gamma = gamma, probability = True, random_state = 0),\
                                      X = self.x, y = self.y, scoring = self.sm, cv = self.folds, n_jobs = self.njobs))

    def randf_cv(self, n_estimators, min_samples_split):
        return numpy.mean(cross_val_score(estimator = RandomForestClassifier(n_estimators = int(n_estimators),\
                                                                          min_samples_split = int(min_samples_split),\
                                                                          random_state = 1),\
                                      X = self.x, y = self.y, scoring = self.sm, cv = self.folds, n_jobs = self.njobs))

    def ensemble(self, c1, c2, c3, pred1, pred2, pred3, y, scoring):
        psum = c1 + c2 + c3
        return scoring(y, (c1/psum)*pred1 + (c2/psum)*pred2 + (c3/psum)*pred3)


    def skf_proba(self, clf_logit, clf_svm, clf_randf):
        
        from sklearn.cross_validation import StratifiedKFold
        skfold = StratifiedKFold(self.y, n_folds = self.folds)
        
        prediction = numpy.zeros((len(self.y), 3))
        
        for itr, ite in skfold:
            xtrain, xtest = self.x[itr], self.x[ite]
            ytrain, ytest = self.y[itr], self.y[ite]

            clf_logit.fit(xtrain, ytrain)
            clf_svm.fit(xtrain, ytrain)
            clf_randf.fit(xtrain, ytrain)
            
            prediction[ite, 0] = clf_logit.predict_proba(xtest)[:, 1]
            prediction[ite, 1] = clf_svm.predict_proba(xtest)[:, 1]
            prediction[ite, 2] = clf_randf.predict_proba(xtest)[:, 1]
            

        return prediction


    def fit(self, x, y):
        '''Try some basic ones and return best + score - like NB, LR, RF, that kind of stuff'''

        self.x = x
        self.y = y


        print('Optimizing Logistic Regression')
        bo_logit = bayes_opt(self.logit_cv, {'C' : (0.001, 100), 'intercept_scaling' : (0.0001, 10)})
        logit_max, self.logit_argmax = bo_logit.log_maximize(restarts = 50, init_points = 5, verbose = 2, num_it = self.it)
        print('Best score found with logit: %f' % logit_max)

        print('Optimizing SVM')
        bo_logit = bayes_opt(self.svm_cv, {'C' : (0.001, 100), 'gamma' : (0.00001, 10)})
        svm_max, self.svm_argmax = bo_logit.log_maximize(restarts = 50, init_points = 5, verbose = 2, num_it = self.it)
        print('Best score found with SVM: %f' % svm_max)

        print('Optimizing Random Forest')
        bo_randf = bayes_opt(self.randf_cv, {'n_estimators' : (10, 200), 'min_samples_split' : (2, 50)})
        randf_max, self.randf_argmax = bo_randf.maximize(restarts = 50, init_points = 5, verbose = 2, num_it = self.it)
        print('Best score found with rf: %f' % randf_max)


        print('Ensemble time')
        if self.sm == 'roc_auc':
            self.logit_model = LogisticRegression(**self.logit_argmax)
            self.svm_model = SVC(**self.svm_argmax)
            self.svm_model.set_params(probability = True, random_state = 0)
            self.randf_model = RandomForestClassifier(n_estimators = int(self.randf_argmax['n_estimators']),\
                                               min_samples_split = int(self.randf_argmax['min_samples_split']),\
                                               random_state = 1)

            pred = self.skf_proba(self.logit_model, self.svm_model, self.randf_model)


            bo_ense = bayes_opt(lambda C1, C2, C3: self.ensemble(c1 = C1, c2 = C2, c3 = C3,\
                                                           pred1 = pred[:, 0],\
                                                           pred2 = pred[:, 1],\
                                                           pred3 = pred[:, 2],\
                                                           y = self.y,\
                                                           scoring = classification_scores[self.sm]),\
                                {'C1' : (1e-15, 1), 'C2' : (1e-15, 1),  'C3' : (1e-15, 1)})
            
            self.coefs = bo_ense.maximize(restarts = 50, init_points = 5, verbose = 2, num_it = 10)
        else:
            print('not done yet')

        print('Fitting the models at last.')

        self.logit_model.fit(self.x, self.y)
        self.svm_model.fit(self.x, self.y)
        self.randf_model.fit(self.x, self.y)


    def predict(self, x):
        '''
        if self.sm == 'roc_auc':
            logit_pred = self.logit_model.predict_proba(x)
            svm_pred = self.svm_model.predict_proba(x)
            randf_pred = self.randf_model.predict_proba(x)

            preds = [logit_pred, svm_pred, randf_pred]

            weights = [self.coefs[key] for key in self.coefs.keys()]
            wsum = reduce(lambda x, y: x+y, weights)
            
            ense = 0
            for i, w in enumerate(weights):
                ense += (w/wsum) * preds[i]
            
            return ense
        '''
        return 0


