#
#   matrix.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from sys import exit


class Matrix(object):
    values: list[list[float]]

    def __init__(self, values: list[list[float]]):
        self.values = values

    def shape(self) -> tuple[int, int]:
        return len(self.values), len(self.values[0])

    def min(self) -> float:
        return min(min(value) for value in self.values)

    def max(self) -> float:
        return max(max(value) for value in self.values)

    def sum(self) -> float:
        return sum(sum(row) for row in self.values)

    def substract(self, matrix: list[list[float]]) -> list[list[float]]:
        return [[i - j for i, j in zip(*m)] for m in zip(self.values, matrix)]

    def multiply(self, matrix: list[list[float]]) -> list[list[float]]:
        if len(self.values[0]) != len(matrix):
            exit("The number of columns in the first matrix must correspond to the number of rows in the second matrix.")
        return [[sum(m * n for m, n in zip(i, j)) for j in zip(*matrix)] for i in self.values]

    def transpose(self) -> list[list[float]]:
        return list(map(list, zip(*self.values)))

    def scalar(self, n: float) -> list[list[float]]:
        return [[x * n for x in row] for row in self.values]

    def mean(self) -> float:
        shape: tuple[int, int] = self.shape()
        return self.sum() / (shape[0] * shape[1])

    def square(self) -> list[list[float]]:
        shape: tuple[int, int] = self.shape()
        return [[self.values[i][j] ** 2 for j in range(shape[1])] for i in range(shape[0])]
