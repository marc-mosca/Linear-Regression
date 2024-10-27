#
#   reader.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from csv import reader
from json import JSONDecodeError, load
from math import isnan
from matrix import Matrix
from sys import exit
from typing import Any, List, Tuple


def reader_csv(path: str) -> Tuple[Matrix, Matrix]:
    target: List[List[float]] = []
    features: List[List[float]] = []

    try:
        with open(path, mode="r", newline="") as file:
            r = reader(file)
            next(r)
            for row in r:
                target.append([float(row[1])])
                features.append([float(row[0])])
        return Matrix(target), Matrix(features)
    except FileNotFoundError:
        exit(f"The file {path} does not exist.")
    except Exception as e:
        exit(f"An error occurred while reading the CSV file: {e}")

def reader_json(path: str) -> Tuple[Matrix, float, float]:
    try:
        with open(path, mode="r") as file:
            data: dict[str, Any] = load(file)
            (theta0, theta1, xmin, xmax) = data.values()
            if isnan(theta0) or isnan(theta1) or isnan(xmin) or isnan(xmax):
                print(f"Invalid float data in JSON, so intializing values to 0.")
                return Matrix([[0.0], [0.0]]), 0.0, 1.0
        return Matrix([[theta0], [theta1]]), xmin, xmax
    except FileNotFoundError:
        print(f"The file {path} was not found, so initializing values to 0.")
        return Matrix([[0.0], [0.0]]), 0.0, 1.0
    except JSONDecodeError:
        exit(f"The file {path} contains invalid JSON.")
    except PermissionError:
        print(f"The file {path} was not readable, so initializing values to 0.")
        return Matrix([[0.0], [0.0]]), 0.0, 1.0
    except Exception as e:
        print(f"Model file {path} doesn't contain the necessary information (theta0, theta1, xmin, xmax).")
        print("So initializing values to 0.")
        return Matrix([[0.0], [0.0]]), 0.0, 1.0
