#
#   matrix.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from sys import exit
from typing import List, Tuple, Union


Numbers = Union[int, float]


class Matrix:
    __list: List[List[Numbers]]
    __rows: int
    __cols: int

    def __init__(self, matrix: List[List[Numbers]]) -> None:
        self.__list = matrix
        self.__rows = 0
        self.__cols = 0
        if len(matrix) != 0:
            self._validate()

    def _validate(self) -> None:
        self.__rows = len(self.__list)
        self.__cols = len(self.__list[0])
        if not all([len(row) == self.__cols for row in self.__list]):
            exit("All rows in the matrix must have equal number of columns.")

    def __str__(self) -> str:
        separator: str = "-" * 9
        if self.__cols == 0:
            return f"{separator}\n{separator}\n{self.shape()}\n"
        matrix_str: str = "\n".join(str(row) for row in self.__list)
        return f"{separator}\n{matrix_str}\n{separator}\n{self.shape()}\n"

    def _zeroed_matrix(self, sizes: Union[Tuple[int, int], None] = None) -> "Matrix":
        if isinstance(sizes, Tuple):
            return Matrix([[0] * sizes[1] for _ in range(sizes[0])])
        else:
            return Matrix([[0] * self.__cols for _ in range(self.__rows)])

    def __add__(self, other: "Matrix") -> "Matrix":
        if self.shape() != other.shape():
            exit("Both matrices must have the same dimensions.")
        matrix: Matrix = self._zeroed_matrix()
        for row in range(self.__rows):
            for column in range(self.__cols):
                matrix.__list[row][column] = self.__list[row][column] + other.__list[row][column]
        return matrix

    def __sub__(self, other: "Matrix") -> "Matrix":
        if self.shape() != other.shape():
            exit("Both matrices must have the same dimensions.")
        matrix: Matrix = self._zeroed_matrix()
        for row in range(self.__rows):
            for column in range(self.__cols):
                matrix.__list[row][column] = self.__list[row][column] - other.__list[row][column]
        return matrix

    def __mul__(self, other: Union["Matrix", Numbers]) -> "Matrix":
        if isinstance(other, (int, float)):
            matrix: Matrix = self._zeroed_matrix()
            for row in range(self.__rows):
                for column in range(self.__cols):
                    matrix.__list[row][column] = self.__list[row][column] * other
            return matrix
        else:
            if self.__cols != other.__rows:
                exit("Number of columns must equal number of rows.")
            matrix: Matrix = self._zeroed_matrix((self.__rows, other.__cols))
            for i in range(self.__rows):
                for j in range(other.__cols):
                    for k in range(self.__cols):
                        matrix.__list[i][j] += self.__list[i][k] * other.__list[k][j]
            return matrix

    def __eq__(self, other: "Matrix") -> bool:
        return self.__list == other.__list

    def get(self) -> List[List[Numbers]]:
        return self.__list

    def min(self) -> float:
        return min(min(row) for row in self.__list)

    def max(self) -> float:
        return max(max(row) for row in self.__list)

    def shape(self) -> Tuple[int, int]:
        return self.__rows, self.__cols

    def sum(self) -> Numbers:
        return sum(sum(row) for row in self.__list)

    def mean(self) -> float:
        return self.sum() / (self.__rows * self.__cols)

    def transpose(self) -> "Matrix":
        matrix: Matrix = self._zeroed_matrix((self.__cols, self.__rows))
        for row in range(self.__rows):
            for column in range(self.__cols):
                matrix.__list[column][row] = self.__list[row][column]
        return matrix

    def square(self) -> "Matrix":
        matrix: Matrix = self._zeroed_matrix()
        for row in range(self.__rows):
            for column in range(self.__cols):
                matrix.__list[row][column] = pow(self.__list[row][column], 2)
        return matrix
