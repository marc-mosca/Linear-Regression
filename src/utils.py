#
#   utils.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from json import dump, load, JSONDecodeError
from matrix import Matrix
from typing import Any


def read_json(path: str) -> tuple[Matrix, float, float]:
    try:
        with open(path, mode="r") as file:
            data: dict[str, Any] = load(file)
        return Matrix([[data["theta0"]], [data["theta1"]]]), data["xmin"], data["xmax"]
    except JSONDecodeError:
        raise RuntimeError(f"The file {path} contains invalid JSON.")
    except Exception as e:
        print(f"Model file {path} not found, initializing theta to 0.")
        return Matrix([[0.0], [0.0]]), 0.0, 1.0


def write_json(path: str, data: dict[str, Any]) -> None:
    try:
        with open(path, mode="w") as file:
            dump(data, file, indent=4)
    except Exception as e:
        raise RuntimeError(f"An error occurred while writing to the JSON file: {e}")
