import numpy as np
from bayes_opt import BayesianOptimization, acquisition
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import matplotlib.pyplot as plt

N_FOLDS = 10
N_START = 2
N_ITER = 25 - N_START
# Load data
data = load_digits()
    

# Define the hyperparameter space
continuous_pbounds = {
    'log_learning_rate': (-10, 0),
    'max_depth': (1, 6),
    'min_samples_split': (2, 6)
}

discrete_pbounds = {
    'log_learning_rate': (-10, 0),
    'max_depth': (1, 6, int),
    'min_samples_split': (2, 6, int)
}

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

res_continuous = []
res_discrete = []

METRIC_SIGN = -1

for i, (train_idx, test_idx) in enumerate(kfold.split(data.data)):
    print(f'Fold {i + 1}/{N_FOLDS}')
    def gboost(log_learning_rate, max_depth, min_samples_split):
        clf = GradientBoostingClassifier(
            n_estimators=10,
            max_depth=int(max_depth),
            learning_rate=np.exp(log_learning_rate),
            min_samples_split=int(min_samples_split),
            random_state=42 + i
        )
        clf.fit(data.data[train_idx], data.target[train_idx])
        #return clf.score(data.data[test_idx], data.target[test_idx])
        return METRIC_SIGN * log_loss(data.target[test_idx], clf.predict_proba(data.data[test_idx]), labels=list(range(10)))
    
    continuous_optimizer = BayesianOptimization(
        f=gboost,
        pbounds=continuous_pbounds,
        acquisition_function=acquisition.ExpectedImprovement(xi=1e-2, random_state=42),
        verbose=0,
        random_state=42,
    )

    discrete_optimizer = BayesianOptimization(
        f=gboost,
        pbounds=discrete_pbounds,
        acquisition_function=acquisition.ExpectedImprovement(xi=1e-2, random_state=42),
        verbose=0,
        random_state=42,
    )
    continuous_optimizer.maximize(init_points=2, n_iter=N_ITER)
    discrete_optimizer.maximize(init_points=2, n_iter=N_ITER)
    res_continuous.append(METRIC_SIGN * continuous_optimizer.space.target)
    res_discrete.append(METRIC_SIGN * discrete_optimizer.space.target)

score_continuous = []
score_discrete = []

for fold in range(N_FOLDS):
    best_in_fold = min(np.min(res_continuous[fold]), np.min(res_discrete[fold]))
    score_continuous.append(np.minimum.accumulate((res_continuous[fold] - best_in_fold)))
    score_discrete.append(np.minimum.accumulate((res_discrete[fold] - best_in_fold)))

mean_continuous = np.mean(score_continuous, axis=0)
quantiles_continuous = np.quantile(score_continuous, [0.1, 0.9], axis=0)
mean_discrete = np.mean(score_discrete, axis=0)
quantiles_discrete = np.quantile(score_discrete, [0.1, 0.9], axis=0)


plt.figure(figsize=(10, 5))
plt.plot((mean_continuous), label='Continuous best seen')
plt.fill_between(range(N_ITER + N_START), quantiles_continuous[0], quantiles_continuous[1], alpha=0.3)
plt.plot((mean_discrete), label='Discrete best seen')
plt.fill_between(range(N_ITER + N_START), quantiles_discrete[0], quantiles_discrete[1], alpha=0.3)

plt.xlabel('Number of iterations')
plt.ylabel('Score')
plt.legend(loc='best')
plt.grid()
plt.savefig('discrete_vs_continuous.png')
