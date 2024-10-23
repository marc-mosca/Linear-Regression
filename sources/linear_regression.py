#
#   linear_regression.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from matrix import Matrix
from typing import Union


def normalize(x: float, xmin: float, xmax: float) -> float:
    return (x - xmin) / (xmax - xmin)

def model(x: Union[Matrix, float], theta: Matrix) -> Union[Matrix, float]:
    if isinstance(x, float):
        theta_value = theta.get()
        return (theta_value[0][0] * x) + theta_value[1][0]
    else:
        return x * theta

def cost_function(x: Matrix, y: Matrix, theta: Matrix) -> float:
    m: int = x.shape()[0]
    return (1 / (2 * m)) * (model(x, theta) - y).square().sum()

def gradient(x: Matrix, y: Matrix, theta: Matrix) -> Matrix:
    m: int = x.shape()[0]
    return x.transpose() * (model(x, theta) - y) * (1 / m)

def gradient_descent(x: Matrix, y: Matrix, theta: Matrix, learning_rate: float, iterations: int) -> Matrix:
    for i in range(iterations):
        theta = theta - (gradient(x, y, theta) * learning_rate)
    return theta

def coefficient_determination(x: Matrix, y: Matrix, theta: Matrix) -> float:
    n: int = y.shape()[0]
    predictions: Matrix = model(x, theta)
    means: Matrix = Matrix([[y.mean()] for _ in range(n)])
    return 1 - ((y - predictions).square().sum() / (y - means).square().sum())
