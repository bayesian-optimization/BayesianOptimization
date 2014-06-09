#from ..bayes_opt.bo.bayes_opt import bayes_opt



## Under construction, nothing to see here.

## This is will soon become a cookie cutter, pre fabricated
## cross validation maker.


class magic_box:


    def __init__(self, x, y, function, score_metric, para_names, n_folds = 3, shuffle = True):

        self.x = x
        self.y = y
        self.f = function
        self.sm = score_metric
        self.para = para_names
        self.folds = n_folds
        self.shuffle = shuffle




    def do(rs = 17):

        from sklearn.cross_validation import cross_val_score
        
        return 0



    def tryall_classification(self):
        '''Try some basic ones and return best + score - like NB, LR, RF, that kind of stuff'''
        
        return 0

    def tryall_regression(self):
        '''Try some basic ones and return best + score - like NB, LR, RF, that kind of stuff'''
        
        return 0
