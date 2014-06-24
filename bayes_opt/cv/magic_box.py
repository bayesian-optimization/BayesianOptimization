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
from sklearn.ensemble import GradientBoostingClassifier


classification_scores = {'accuracy' : metrics.accuracy_score,\
                        'average_precision' : metrics.average_precision_score,\
                        'f1' : metrics.f1_score,\
                        'precision' : metrics.precision_score,\
                        'recall' : metrics.recall_score,\
                        'roc_auc' : metrics.roc_auc_score,}



class magic_box_classifier:


    # --------------------------------------------- // --------------------------------------------- #
    # --------------------------------------------- // --------------------------------------------- # 
    def __init__(self, score_metric, n_folds = 10, n_jobs = 1, iterations = 50):      

        self.sm = score_metric
        self.folds = n_folds
        self.njobs = n_jobs
        self.it = iterations

        self.logit_argmax = {'C' : 1, 'intercept_scaling' : 1}
        self.svm_argmax = {'C' : 1, 'gamma' : 0.0}
        self.randf_argmax = {'n_estimators' : 10, 'min_samples_split' : 2, 'min_samples_leaf' : 1,\
                             'max_depth' : 3, 'max_features' : 'sqrt'}
        self.gbt_argmax = {'n_estimators' : 10, 'min_samples_split' : 2, 'min_samples_leaf' : 1,\
                           'max_depth' : 3, 'max_features' : 'sqrt'}

        self.coefs = {'c_logit' : 0.25, 'c_svm' : 0.25, 'c_randf' : 0.25, 'c_gbt' : 0.25}

        self.logit_model = 0
        self.svm_model = 0
        self.randf_model = 0
        self.gbt_model = 0


    # --------------------------------------------- // --------------------------------------------- # 
    def get_parameters(self):

        models = ['logit', 'svm', 'rf', 'gbt']
        paras = [self.logit_argmax, self.svm_argmax, self.randf_argmax,  self.gbt_argmax]

        return dict(zip(models, paras))

    def get_coefficients(self):

        return self.coefs 

    # --------------------------------------------- // --------------------------------------------- # 
    def logit_cv(self, C, intercept_scaling):
        return numpy.mean(cross_val_score(estimator = LogisticRegression(C = C, intercept_scaling = intercept_scaling),\
                                      X = self.x, y = self.y, scoring = self.sm, cv = self.folds, n_jobs = self.njobs))

    def svm_cv(self, C, gamma):
        return numpy.mean(cross_val_score(estimator = SVC(C = C, gamma = gamma, probability = True, random_state = 0),\
                                      X = self.x, y = self.y, scoring = self.sm, cv = self.folds, n_jobs = self.njobs))

    def randf_cv(self, n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features):
        
        estima = RandomForestClassifier(n_estimators = int(n_estimators),\
                                       min_samples_split = int(min_samples_split),\
                                       min_samples_leaf = int(min_samples_leaf),\
                                       max_depth= int(max_depth),\
                                       max_features = max_features,\
                                       random_state = 1)
        
        return numpy.mean(cross_val_score(estimator = estima,\
                                        X = self.x,\
                                        y = self.y,\
                                        scoring = self.sm,\
                                        cv = self.folds,\
                                        n_jobs = self.njobs))


    def gbt_cv(self, n_estimators, min_samples_split, min_samples_leaf, max_depth, max_features):

        estima = GradientBoostingClassifier(n_estimators = int(n_estimators),\
                                          min_samples_split = int(min_samples_split),\
                                          min_samples_leaf = int(min_samples_leaf),\
                                          max_depth= int(max_depth),\
                                          max_features = max_features,\
                                          random_state = 2)
        
        return numpy.mean(cross_val_score(estimator = estima,\
                                        X = self.x,\
                                        y = self.y,\
                                        scoring = self.sm,\
                                        cv = self.folds,\
                                        n_jobs = self.njobs))

    # --------------------------------------------- // --------------------------------------------- # 
    def ensemble(self, c1, c2, c3, c4, pred1, pred2, pred3, pred4, y, scoring):
        psum = c1 + c2 + c3 + c4
        return scoring(y, (c1/psum)*pred1 + (c2/psum)*pred2 + (c3/psum)*pred3 + (c4/psum)*pred4)

    # --------------------------------------------- // --------------------------------------------- # 
    def skf_proba(self, clf_logit, clf_svm, clf_randf, clf_gbt):
        
        from sklearn.cross_validation import StratifiedKFold
        skfold = StratifiedKFold(self.y, n_folds = self.folds)
        
        prediction = numpy.zeros((len(self.y), 4))
        
        for itr, ite in skfold:
            xtrain, xtest = self.x[itr], self.x[ite]
            ytrain, ytest = self.y[itr], self.y[ite]

            clf_logit.fit(xtrain, ytrain)
            clf_svm.fit(xtrain, ytrain)
            clf_randf.fit(xtrain, ytrain)
            clf_gbt.fit(xtrain, ytrain)
            
            prediction[ite, 0] = clf_logit.predict_proba(xtest)[:, 1]
            prediction[ite, 1] = clf_svm.predict_proba(xtest)[:, 1]
            prediction[ite, 2] = clf_randf.predict_proba(xtest)[:, 1]
            prediction[ite, 3] = clf_gbt.predict_proba(xtest)[:, 1]
            

        return prediction


    def skf(self, clf_logit, clf_svm, clf_randf, clf_gbt):
        
        from sklearn.cross_validation import StratifiedKFold
        skfold = StratifiedKFold(self.y, n_folds = self.folds)
        
        prediction = numpy.zeros((len(self.y), 4))
        
        for itr, ite in skfold:
            xtrain, xtest = self.x[itr], self.x[ite]
            ytrain, ytest = self.y[itr], self.y[ite]

            clf_logit.fit(xtrain, ytrain)
            clf_svm.fit(xtrain, ytrain)
            clf_randf.fit(xtrain, ytrain)
            clf_gbt.fit(xtrain, ytrain)
            
            prediction[ite, 0] = clf_logit.predict(xtest)
            prediction[ite, 1] = clf_svm.predict(xtest)
            prediction[ite, 2] = clf_randf.predict(xtest)
            prediction[ite, 3] = clf_gbt.predict(xtest)

        return prediction

    # --------------------------------------------- // --------------------------------------------- # 
    def model_opt(self, model, params, name, log_max = False):

        print('Optimizing %s' % name)
        
        bo = bayes_opt(model, params)
        if log_max:
            max_val, argmax = bo.log_maximize(restarts = 100, init_points = 10, verbose = 1, num_it = self.it)
        else:
            max_val, argmax = bo.maximize(restarts = 100, init_points = 10, verbose = 1, num_it = self.it)

        print('Best score found with %s: %f' % (name, max_val))

        #Return intergers for trees
        if 'n_estimators' in params.keys():
            argmax['n_estimators'] = int(argmax['n_estimators'])
            argmax['min_samples_split'] = int(argmax['min_samples_split'])
            argmax['min_samples_leaf'] = int(argmax['min_samples_leaf'])
            argmax['max_depth'] = int(argmax['max_depth'])

        return argmax

    # --------------------------------------------- // --------------------------------------------- # 
    def best_args(self, x, y):

        self.x = x
        self.y = y

        logit_params = {'C' : (0.0001, 100), 'intercept_scaling' : (0.0001, 10)}
        svm_params = {'C' : (0.0001, 100), 'gamma' : (0.00001, 10)}
        randf_params = {'n_estimators' : (10, 1000),\
                        'min_samples_split' : (2, 200),\
                        'min_samples_leaf' : (1, 100),\
                        'max_depth' : (1, 20),\
                        'max_features' : (0.1, 0.999)}
        gbt_params = {'n_estimators' : (10, 1000),\
                      'min_samples_split' : (2, 200),\
                      'min_samples_leaf' : (1, 100),\
                      'max_depth' : (1, 20),\
                      'max_features' : (0.1, 0.999)}

        self.logit_argmax = self.model_opt(self.logit_cv, logit_params, 'Logistic Regression', log_max = True)
        self.svm_argmax = self.model_opt(self.svm_cv, svm_params, 'SVM', log_max = True)
        self.randf_argmax = self.model_opt(self.randf_cv, randf_params, 'Random Forest')
        self.gbt_argmax = self.model_opt(self.gbt_cv, gbt_params, 'Gradient Boosted Trees')


    # --------------------------------------------- // --------------------------------------------- # 
    def make_ensemble(self):

        print('Ensemble time')
        self.logit_model = LogisticRegression(**self.logit_argmax)
        self.svm_model = SVC(**self.svm_argmax)
        self.randf_model = RandomForestClassifier(**self.randf_argmax)
        self.gbt_model = GradientBoostingClassifier(**self.gbt_argmax)

        self.svm_model.set_params(random_state = 0)
        self.randf_model.set_params(random_state = 1)
        self.gbt_model.set_params(random_state = 2)

        coefs_range = {'c_logit' : (1e-15, 1), 'c_svm' : (1e-15, 1),  'c_randf' : (1e-15, 1),  'c_gbt' : (1e-15, 1)}
        
        if self.sm == 'roc_auc':
            self.svm_model.set_params(probability = True)
            
            pred = self.skf_proba(self.logit_model, self.svm_model, self.randf_model, self.gbt_model)
            bo_ense = bayes_opt(lambda c_logit, c_svm, c_randf, c_gbt: self.ensemble(\
                                                                     c1 = c_logit,\
                                                                     c2 = c_svm,\
                                                                     c3 = c_randf,\
                                                                     c4 = c_gbt,\
                                                                     pred1 = pred[:, 0],\
                                                                     pred2 = pred[:, 1],\
                                                                     pred3 = pred[:, 2],\
                                                                     pred4 = pred[:, 3],\
                                                                     y = self.y,\
                                                                     scoring = classification_scores[self.sm]), coefs_range)
            
            best_ensemble, self.coefs = bo_ense.maximize(restarts = 100, init_points = 5, verbose = 1, num_it = 50)
            
        else:
            pred = self.skf(self.logit_model, self.svm_model, self.randf_model, self.gbt_model)
            bo_ense = bayes_opt(lambda c_logit, c_svm, c_randf, c_gbt: self.ensemble(\
                                                                     c1 = c_logit,\
                                                                     c2 = c_svm,\
                                                                     c3 = c_randf,\
                                                                     c4 = c_gbt,\
                                                                     pred1 = pred[:, 0],\
                                                                     pred2 = pred[:, 1],\
                                                                     pred3 = pred[:, 2],\
                                                                     pred4 = pred[:, 3],\
                                                                     y = self.y,\
                                                                     scoring = classification_scores[self.sm]), coefs_range)
            
            best_ensemble, self.coefs = bo_ense.maximize(restarts = 100, init_points = 5, verbose = 1, num_it = 50)

            print('Best ensemble score: %f' % best_ensemble)


# --------------------------------------------- // --------------------------------------------- #
# --------------------------------------------- // --------------------------------------------- # 
    def fit(self, x, y):
        '''Try some basic ones and return best + score - like NB, LR, RF, that kind of stuff'''


        self.best_args(x, y)

        self.make_ensemble()


        print('Fitting the models at last.')

        self.logit_model.fit(self.x, self.y)
        self.svm_model.fit(self.x, self.y)
        self.randf_model.fit(self.x, self.y)
        self.gbt_model.fit(self.x, self.y)

        print('Done, best single model and parameters...')


# --------------------------------------------- // --------------------------------------------- #
# --------------------------------------------- // --------------------------------------------- # 
    def predict_proba(self, x):
        print('starting prob prediction')

        logit_pred = self.logit_model.predict_proba(x)
        svm_pred = self.svm_model.predict_proba(x)
        randf_pred = self.randf_model.predict_proba(x)
        gbt_pred = self.gbt_model.predict_proba(x)


        wsum = self.coefs['c_logit'] + self.coefs['c_svm'] + self.coefs['c_randf'] + self.coefs['c_gbt']
  
        ense = (self.coefs['c_logit']/wsum) * logit_pred +\
               (self.coefs['c_svm']/wsum) * svm_pred +\
               (self.coefs['c_randf']/wsum) * randf_pred +\
               (self.coefs['c_gbt']/wsum) * gbt_pred
            
        return ense

'''
    def predict(self, x):
        print('starting prediction')

        logit_pred = self.logit_model.predict(x)
        svm_pred = self.svm_model.predict(x)
        randf_pred = self.randf_model.predict(x)


        wsum = self.coefs['c_logit'] + self.coefs['c_svm'] + self.coefs['c_randf']
  
        ense = (self.coefs['c_logit']/wsum) * logit_pred +\
               (self.coefs['c_svm']/wsum) * svm_pred +\
               (self.coefs['c_randf']/wsum) * randf_pred
            
        return ense
'''


