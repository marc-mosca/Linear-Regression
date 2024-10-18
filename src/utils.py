#
#   utils.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from json import dump, load, JSONDecodeError
from matrix import Matrix
from os import makedirs, path
from sys import exit
from typing import Any


def read_json(filepath: str) -> tuple[Matrix, float, float]:
    try:
        with open(filepath, mode="r") as file:
            data: dict[str, Any] = load(file)
        return Matrix([[data["theta0"]], [data["theta1"]]]), data["xmin"], data["xmax"]
    except FileNotFoundError:
        print(f"The file {filepath} was not found, so initializing theta to 0.")
        return Matrix([[0.0], [0.0]]), 0.0, 1.0
    except JSONDecodeError:
        exit(f"The file {filepath} contains invalid JSON.")
    except PermissionError:
        print(f"The file {filepath} was not readable, so initializing theta to 0.")
        return Matrix([[0.0], [0.0]]), 0.0, 1.0
    except Exception as e:
        print(f"Model file {filepath} doesn't contain the necessary information (theta0, theta1, xmin, xmax), so initializing theta to 0.")
        return Matrix([[0.0], [0.0]]), 0.0, 1.0


def write_json(filepath: str, data: dict[str, Any]) -> None:
    try:
        directory: str = path.dirname(filepath)
        if not path.exists(directory):
            makedirs(directory)
        with open(filepath, mode="w") as file:
            dump(data, file, indent=4)
    except Exception as e:
        exit(f"An error occurred while writing to the JSON file: {e}")
