import numpy as np
from scipy.optimize import minimize

A = 10
a = 3
b = 1

f = lambda x: A - (x[0]-a)*np.exp(-(x[0]-a)) - (x[1]-b)*np.exp(-(x[1]-b))

x = np.array([0.0, 0.0])

eps = 1e-6

# Сопряженные направления
S = np.eye(2)

while True:
    x_old = x.copy()
    for i in range(2):
        result = minimize(lambda lam: f(x + lam * S[:, i]), 0)
        x = x + result.x * S[:, i]

    y = x - x_old
    if np.linalg.norm(y) < eps:
        break
    
    # Обновление сопряженных направлений
    S = np.roll(S, -1, axis=1)
    S[:, -1] = y / np.linalg.norm(y)

print(x)
