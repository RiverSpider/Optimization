import numpy as np

# Параметры задачи
A = 10
a = 3
b = 1

# Функция, которую нужно оптимизировать
def f(x):
    return A - (x[0] - a) * np.exp(-(x[0] - a)) - (x[1] - b) * np.exp(-(x[1] - b))

# Шаг оптимизации
alpha = 0.001

# Критерий остановки
epsilon = 1e-6

# Начальное приближение
x = np.array([0.0, 0.0])

# Метод Розенброка
while True:
    x_prev = x.copy()

    # Поиск вдоль оси x
    while True:
        f_prev = f(x)
        x[0] -= alpha * ((2 * (x[0] - a) * np.exp(-(x[0] - a)) * (1 + (x[1] - a))) + ((x[1] - b) * np.exp(-(x[0] - b))))
        if abs(f(x) - f_prev) < epsilon:
            break

    # Поиск вдоль оси y
    while True:
        f_prev = f(x)
        x[1] -= alpha * (x[1] - b) * np.exp(-(x[0] - b))
        if abs(f(x) - f_prev) < epsilon:
            break

    # Проверка критерия остановки
    if np.linalg.norm(x - x_prev) < epsilon:
        break

print("Optimal x:", x[0])
print("Optimal y:", x[1])
