from __future__ import print_function
from __future__ import division

from sklearn.datasets import make_classification
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

from bayes_opt import BayesianOptimization

# Load data set and target values
data, target = make_classification(
    n_samples=1000,
    n_features=45,
    n_informative=12,
    n_redundant=7
)

def svccv(C, gamma):
    val = cross_val_score(
        SVC(C=C, gamma=gamma, random_state=2),
        data, target, 'f1', cv=2
    ).mean()

    return val

def rfccv(n_estimators, min_samples_split, max_features):
    val = cross_val_score(
        RFC(n_estimators=int(n_estimators),
            min_samples_split=int(min_samples_split),
            max_features=min(max_features, 0.999),
            random_state=2
        ),
        data, target, 'f1', cv=2
    ).mean()
    return val

if __name__ == "__main__":
    gp_params = {"alpha": 1e-5}

    svcBO = BayesianOptimization(svccv,
        {'C': (0.001, 100), 'gamma': (0.0001, 0.1)})
    svcBO.explore({'C': [0.001, 0.01, 0.1], 'gamma': [0.001, 0.01, 0.1]})

    rfcBO = BayesianOptimization(
        rfccv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999)}
    )

    # Add a constraint that we have five times as many estimators as features.
    rfcBO_constr = BayesianOptimization(
        rfccv,
        {'n_estimators': (10, 250),
        'min_samples_split': (2, 25),
        'max_features': (0.1, 0.999)},
        constraints=[{'type': 'ineq',
                      'fun': lambda x: int(x[0]) - 5*int(x[2] * 45)}]
    )

    svcBO.maximize(n_iter=10, **gp_params)
    print('-' * 53)
    rfcBO.maximize(n_iter=10, **gp_params)
    print('-' * 53)
    rfcBO_constr.maximize(n_iter=10, **gp_params)
    print('-' * 53)
    print('Final Results')
    print('SVC: %f' % svcBO.res['max']['max_val'])
    print('RFC: %f' % rfcBO.res['max']['max_val'])
    print(f"RFC with constraints: {rfcBO_constr.res['max']['max_val']}")
