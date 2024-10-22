#
#   writer.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from json import dump
from os import makedirs, path
from sys import exit
from typing import Any


def writer_json(to: str, data: dict[str, Any]) -> None:
    try:
        directory: str = path.dirname(to)
        if not path.exists(directory):
            makedirs(directory)

        with open(to, mode="w") as file:
            dump(data, file, indent=4)
    except Exception as e:
        exit(f"An error occurred while writing to the JSON file: {e}")
