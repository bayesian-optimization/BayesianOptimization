from ..bo.bayes_opt import bayes_opt
import numpy

from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier



## Under construction, nothing to see here.

## This is will soon become a cookie cutter, pre fabricated
## cross validation maker.


class magic_box_classifier:

   


    def __init__(self, score_metric, n_folds = 5, n_jobs = 1):      

        self.sm = score_metric
        self.folds = n_folds
        self.njobs = n_jobs

        self.logit_argmax = {'C' : 1, 'intercept_scaling' : 1}
        self.randf_argmax = {'n_estimators' : 10, 'min_samples_split' : 2}

        self.c_logit = 0.5
        self.c_randf = 0.5


    def logit_cv(self, C, intercept_scaling):
        return numpy.mean(cross_val_score(estimator = LogisticRegression(C = C, intercept_scaling = intercept_scaling),\
                                      X = self.x, y = self.y, scoring = self.sm, cv = self.folds, n_jobs = self.njobs))

    def randf_cv(self, n_estimators, min_samples_split):
        return numpy.mean(cross_val_score(estimator = RandomForestClassifier(n_estimators = int(n_estimators),\
                                                                        min_samples_split = int(min_samples_split)),\
                                      X = self.x, y = self.y, scoring = self.sm, cv = self.folds, n_jobs = self.njobs))

    def ensemble(self, c1, c2, pred1, pred2):
        psum = c1 + c2
        return (c1/psum)*pred1 + (c2/psum)*pred2



    def fit(self, x, y):
        '''Try some basic ones and return best + score - like NB, LR, RF, that kind of stuff'''

        self.x = x
        self.y = y

        logit = {'C' : (0.001, 100), 'intercept_scaling' : (0.0001, 10)}
        randf = {'n_estimators' : (10, 200), 'min_samples_split' : (2, 50)}

        print('Optimizing Logistic Regression')
        bo_logit = bayes_opt(self.logit_cv, {'C' : (0.001, 100), 'intercept_scaling' : (0.0001, 10)})
        logit_max, self.logit_argmax = bo_logit.log_maximize(restarts = 50, init_points = 5, verbose = 2, num_it = 5)
        print('Best score found with logit: %f' % logit_max)


        print('Optimizing Random Forest')
        bo_randf = bayes_opt(self.randf_cv, {'n_estimators' : (10, 200), 'min_samples_split' : (2, 50)})
        randf_max, self.randf_argmax = bo_randf.maximize(restarts = 50, init_points = 5, verbose = 2, num_it = 5)
        print('Best score found with rf: %f' % randf_max)

        print('Ensemble time')

        #clf_logit = LogisticRegression(**self.logit_argmax)
        #pred_logit = clf_logit.predict


    def predict(self, x):
        return 0


'''
class bayes_opt_cv:

    def __init__(self):



    def do(self, para_names, function, rs = 17):

        self.para = para_names
        self.f = function

        from sklearn.cross_validation import cross_val_score
        
        return 0
'''
