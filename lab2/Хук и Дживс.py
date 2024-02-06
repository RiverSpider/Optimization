import numpy as np

A = 10
a = 3
b = 1
f = lambda x: A - (x[0]-a)*np.exp(-(x[0]-a)) - (x[1]-b)*np.exp(-(x[1]-b))

x_start = np.array([0.0, 0.0]) 
delta = 0.01
epsilon = 1e-6

x = x_start.copy()
x_new = x_start.copy()
while True:
    # Исследовательский поиск
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += delta
        x_minus = x.copy()
        x_minus[i] -= delta
        if f(x_plus) < f(x):
            x_new[i] = x_plus[i]
        elif f(x_minus) < f(x):
            x_new[i] = x_minus[i]
    # Узловой поиск
    x_pattern = 2*x_new - x
    if f(x_pattern) < f(x):
        x = x_pattern.copy()
    else:
        if np.linalg.norm(delta) < epsilon:
            break
        delta /= 2
        x_new = x.copy()

print(x)
