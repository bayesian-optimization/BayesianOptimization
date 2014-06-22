from ..bo.bayes_opt import bayes_opt



## Under construction, nothing to see here.

## This is will soon become a cookie cutter, pre fabricated
## cross validation maker.


class magic_box:


    def __init__(self, x, y, function, score_metric, , n_folds = 5):

        self.x = x
        self.y = y
        self.f = function
        self.sm = score_metric
        
        self.para = 'no_name'
        self.folds = n_folds




    def do(self, para_names, rs = 17):

        self.para = para_names

        from sklearn.cross_validation import cross_val_score
        
        return 0



    def tryall(self, prob_type = 'classification', n_jobs = 1, ensemble = True):
        '''Try some basic ones and return best + score - like NB, LR, RF, that kind of stuff'''

        from sklearn.cross_validation import cross_val_score

        if prob_type = 'classification':
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier


            logit = {'c' : (0.001, 100), 'ints' : (0.0001, 10)}
            randf = {'trees' : (10, 200), 'split' : (2, 50)}

            def logit_cv(c, g):
                return numpy.mean(cross_val_score(estimator = LogisticRegression(C = c, intercept_scaling = ints),\
                                      X = self.x, y = self.y, scoring = self.sm, cv = self.folds, n_jobs = n_jobs))

            def randf_cv(trees, split):
                return numpy.mean(cross_val_score(estimator = RandomForestClassifier(n_estimators = trees, min_samples_split = split),\
                                      X = self.x, y = self.y, scoring = self.sm, cv = self.folds, n_jobs = n_jobs))



            bo_logit = bayes_opt(logit_cv, {'c' : (0.001, 100), 'ints' : (0.0001, 10)})
            logit_max, logit_argmax = bo_logit.log_maximize(restarts = 50, init_points = 5, verbose = 0, num_it = 50)


            bo_randf = bayes_opt(randf_cv, {'trees' : (10, 200), 'split' : (2, 50)})
            randf_max, randf_argmax = bo_randf.maximize(restarts = 50, init_points = 5, verbose = 0, num_it = 50)

            if ensemble:
                return 0

            else:



        else:
            print('Regression under construction.')
            return 0
        
        return 0



if __name__ = '__main__':
    from sklearn import datasets
    
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    Y = iris.target
