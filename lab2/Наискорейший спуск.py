import numpy as np
from scipy.optimize import minimize_scalar

A = 10
a = 3
b = 1
f = lambda x: A - (x[0]-a)*np.exp(-(x[0]-a)) - (x[1]-b)*np.exp(-(x[1]-b))

def grad_f(x):
    dfdx = (x[0]-a)*np.exp(-(x[0]-a)) + 1
    dfdy = (x[1]-b)*np.exp(-(x[1]-b)) + 1
    return np.array([dfdx, dfdy])

x = np.array([0.0, 0.0])
eps = 1e-6

while True:
    g = grad_f(x)
    
    if np.linalg.norm(g) < eps:
        break

    # направление спуска
    S = -g / np.linalg.norm(g)

    res = minimize_scalar(lambda alpha: f(x + alpha*S))
    alpha_opt = res.x


    x = x + alpha_opt * S

print(x)
