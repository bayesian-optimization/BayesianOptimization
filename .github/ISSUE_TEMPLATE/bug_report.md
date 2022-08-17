---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug, enhancement
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.
Ex: Using `scipy==1.8` with `bayesian-optimization==1.2.0` results in `TypeError: 'float' object is not subscriptable`.



**To Reproduce**
A concise, self-contained code snippet that reproduces the bug you would like to report.

Ex:
```python
from bayes_opt import BayesianOptimization

black_box_function = lambda x, y: -x ** 2 - (y - 1) ** 2 + 1

pbounds = {'x': (2, 4), 'y': (-3, 3)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds
)
optimizer.maximize()
```

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
 - OS: [e.g. Arch Linux, macOS, Windows]
 - `python` Version [e.g. 3.8.9]
 - `numpy` Version [e.g. 1.21.6]
 - `scipy` Version [e.g. 1.8.0]
 - `bayesian-optimization` Version [e.g. 1.2.0]

**Additional context**
Add any other context about the problem here.
