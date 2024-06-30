"""Originally by @rhizhiy
https://github.com/bayesian-optimization/BayesianOptimization/issues/347#issuecomment-1273465096
"""
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from bayes_opt import BayesianOptimization
from bayes_opt import acquisition
from scipy.optimize import rosen

ASYNC_METHOD = 'lie_max'
assert ASYNC_METHOD in ['lie_max', 'none']
def _closest_distance(point, points):
    return min(np.linalg.norm(point - p) for p in points if p is not point)


def optimize(
    func: Callable[..., float], num_iter: int, bounds: dict[str, tuple[float, float]], num_workers=0
):
    init_samples = int(np.sqrt(num_iter))
    init_kappa = 10
    kappa_decay = (0.1 / init_kappa) ** (1 / num_iter)

    acquisition_function = acquisition.UpperConfidenceBound(
        kappa=init_kappa,
        exploration_decay=kappa_decay,
        exploration_decay_delay=0
    )

    if ASYNC_METHOD == 'lie_max':
        acquisition_function = acquisition.ConstantLiar(acquisition_function, 'max')


    optimizer = BayesianOptimization(
        f=None,
        acquisition_function=acquisition_function,
        pbounds=bounds,
        verbose=0
    )


    init_queue = [optimizer.suggest() for _ in range(init_samples)]
    result_queue = []
    while len(optimizer.res) < num_iter:
        sample = init_queue.pop(0) if init_queue else optimizer.suggest()
        loss = func(list(sample.values())) * -1
        result_queue.append((sample, loss))
        if len(result_queue) >= num_workers:
            optimizer.register(*result_queue.pop(0))
    return optimizer.res


bounds = {"x": [-5, 5], "y": [-5, 5]}

all_times = {}
all_results = {}
workers_each = [1, 2, 4, 8,]# 16]
print(f"Simulating parallel optimization for {workers_each} workers, this can take some time.")
print(f"Async method: {ASYNC_METHOD}.")
for num_workers in workers_each:
    print(f"\tChecking {num_workers} workers")
    results = []
    start = time.perf_counter()
    results = optimize(rosen, 200, bounds, num_workers)
    end = time.perf_counter()
    delta = end - start
    all_times[num_workers] = delta
    samples = [res["params"] for res in results]
    all_results[num_workers] = samples

fig, axs = plt.subplots(2, 2)
if ASYNC_METHOD == 'lie_max':
    acquisition_function_str = "Constant Max Liar (UCB)"
else:
    acquisition_function_str = "UCB"

fig.suptitle(f"Acquisition function: {acquisition_function_str}")
fig.set_figheight(8)
fig.set_figwidth(8)
axs = [item for sublist in axs for item in sublist]
for idx, (num_workers, samples) in enumerate(all_results.items()):
    if num_workers > 8:
        continue
    samples = [np.array(list(sample.values())) for sample in samples]
    axs[idx].scatter(*zip(*samples), s=1)
    axs[idx].set_title(f"{num_workers=}")
    avg_min_distance = np.mean([_closest_distance(sample, samples) for sample in samples])
    print(f"{num_workers=}, mean_min_distance={avg_min_distance:.3f}, time={all_times[num_workers]:.3f}")
fig.tight_layout()
plt.savefig(f"corrected_async_{ASYNC_METHOD}.png")
