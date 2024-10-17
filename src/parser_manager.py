#
#   parser_manager.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from config import DATASET_PATH, TRAINED_MODEL_PATH
from sys import argv, exit
from os import path, access, R_OK


class ParserManager(object):
    arguments: list[str]
    flag: str
    path: str

    def __init__(self):
        self.arguments = argv[1:]
        self.flag = ""
        self.path = ""

    def parse(self) -> None:
        length: int = len(self.arguments)

        if length < 1 or length > 2:
            exit("One or two arguments are required: a flag and an optional file path.")

        self.flag = self.arguments[0]

        if self.flag not in ["--training", "--prediction", "--bonus"]:
            exit(f"Invalid flag '{self.flag}'. Valid flags are '--training', '--prediction', '--bonus'.")

        if length != 2:
            self.path = TRAINED_MODEL_PATH if self.flag == "--prediction" else DATASET_PATH
        else:
            self.path = self.arguments[1]

            if not (self.path.endswith(".csv") or self.path.endswith(".json")):
                exit(f"Invalid file type '{self.path}'. Only .csv and .json files are allowed.")
            elif not path.isfile(self.path):
                exit(f"File '{self.path}' does not exist.")
            elif not access(self.path, R_OK):
                exit(f"File '{self.path}' is not readable.")
