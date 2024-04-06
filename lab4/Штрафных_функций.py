import numpy as np

def objective_function(x, y):
    return x**2 - 2*x + y**2 - y

def inequality_constraint_1(x, y):
    return x**2 - y

def inequality_constraint_2(x, y):
    return y - 5

def penalty_function(x, y, c):
    return objective_function(x, y) + c * max(0, inequality_constraint_1(x, y))**2 + c * max(0, inequality_constraint_2(x, y))**2

def gradient_objective_function(x, y):
    return np.array([2*x - 2, 2*y - 1])

def gradient_penalty_function(x, y, c):
    return gradient_objective_function(x, y) + 2 * c * np.array([max(0, inequality_constraint_1(x, y)) * x, max(0, inequality_constraint_2(x, y))])

def backtracking_line_search(x, y, d, c):
    alpha = 1
    while penalty_function(x + alpha * d[0], y + alpha * d[1], c) > penalty_function(x, y, c) - 0.5 * alpha * np.linalg.norm(gradient_penalty_function(x, y, c))**2:
        alpha /= 2
    return alpha

def method_of_penalty_functions(x0, y0, epsilon):
    x = x0
    y = y0
    c = 1
    
    while True:
        d = -gradient_penalty_function(x, y, c)
        alpha = backtracking_line_search(x, y, d, c)
        if np.linalg.norm(alpha * d) < epsilon:
            break
        x += alpha * d[0]
        y += alpha * d[1]
        c *= 2
    
    return x, y

x0 = 2
y0 = 3
epsilon = 0.05
                        
x, y = method_of_penalty_functions(x0, y0, epsilon)
print(f"x = {x}, y = {y}")
print(f"f = {x**2 - 2*x + y**2 - y}")
x = 0.9735107421875, y = 0.939453125
f = -1.0561792701482773
