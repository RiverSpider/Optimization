import numpy as np
from scipy.optimize import minimize_scalar

A = 10
a = 3
b = 1
f = lambda x: A - (x[0]-a)*np.exp(-(x[0]-a)) - (x[1]-b)*np.exp(-(x[1]-b))

eps = 1e-6
x0 = np.array([0, 0])
n = 2  # размерность пространства

y = np.copy(x0)
k = 1
j = 1

while True:
    # Одномерный поиск
    res = minimize_scalar(lambda lmbd: f(y + lmbd * np.eye(n)[j-1]))
    lmbd = res.x

    # Обновление y
    y = y + lmbd * np.eye(n)[j-1]

    if j < n:
        j += 1
    else:
        if np.linalg.norm(y - x0) < eps:
            break
        else:
            x0 = y
            j = 1
            k += 1

print(y)
