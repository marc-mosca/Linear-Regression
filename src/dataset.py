#
#   dataset.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from csv import reader
from matrix import Matrix


class Dataset(object):
    target: Matrix
    features: Matrix

    def __init__(self, path: str):
        self.target = Matrix([])
        self.features = Matrix([])
        self.__read_dataset(path)

    def __read_dataset(self, path: str) -> None:
        try:
            with open(path, mode="r", newline="") as file:
                r = reader(file)
                next(r)
                for row in r:
                    self.target.values.append([float(row[1])])
                    self.features.values.append([float(row[0])])
        except FileNotFoundError:
            raise RuntimeError(f"The file {path} does not exist.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the CSV file: {e}")
