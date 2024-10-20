#
#   linear_regression.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from matrix import Matrix


class LinearRegression(object):
    iterations: int
    learning_rate: float

    def __init__(self, iterations: int, learning_rate: float):
        self.iterations = iterations
        self.learning_rate = learning_rate

    def normalize(self, x: float, xmin: float, xmax: float) -> float:
        return (x - xmin) / (xmax - xmin)

    def model(self, x: float, theta: Matrix) -> float:
        return (theta.values[0][0] * x) + theta.values[1][0]

    def matrix_model(self, x: Matrix, theta: Matrix) -> Matrix:
        return Matrix(x.multiply(theta.values))

    def cost_function(self, x: Matrix, y: Matrix, theta: Matrix) -> float:
        m: int = x.shape()[0]
        predictions: Matrix = self.matrix_model(x, theta)
        errors: Matrix = Matrix(predictions.substract(y.values))
        squared: Matrix = Matrix(errors.square())
        return 1 / (2 * m) * squared.sum()

    def gradient(self, x: Matrix, y: Matrix, theta: Matrix) -> Matrix:
        m: int = x.shape()[0]
        predictions: Matrix = self.matrix_model(x, theta)
        errors: Matrix = Matrix(predictions.substract(y.values))
        x_t: Matrix = Matrix(x.transpose())
        grad: Matrix = Matrix(x_t.multiply(errors.values))
        return Matrix([[element * (1 / m) for element in row] for row in grad.values])

    def gradient_descent(self, x: Matrix, y: Matrix, theta: Matrix) -> Matrix:
        for i in range(self.iterations):
            gradient: Matrix = self.gradient(x, y, theta)
            updated_theta: Matrix = Matrix(gradient.scalar(self.learning_rate))
            theta: Matrix = Matrix(theta.substract(updated_theta.values))
        return theta

    def coefficient_determination(self, y: Matrix, predictions: Matrix) -> float:
        n: int = y.shape()[0]
        errors: Matrix = Matrix(predictions.substract(y.values))
        squared_errors: Matrix = Matrix(errors.square())
        means: Matrix = Matrix([[y.mean()] for _ in range(n)])
        deviations: Matrix = Matrix(y.substract(means.values))
        squared_deviations: Matrix = Matrix(deviations.square())
        return 1 - (squared_errors.sum() / squared_deviations.sum())
