#
#   parser_manager.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from sys import argv
from os import path, access, R_OK


class ParserManager(object):
    """
    A class to manage and parse command-line arguments for the program. It validates
    a required flag and an optional file path.

    Attributes
    ----------
    arguments : list[str]
        A list of command-line arguments provided to the program.
    flag : str
        The flag argument parsed from the command line, representing the mode of operation.
    path : str
        The file path argument parsed from the command line, if provided.
    """

    arguments: list[str]
    flag: str
    path: str

    def __init__(self) -> None:
        """
        Initializes the ParserManager instance. Extracts command-line arguments
        from the global sys.argv list, skipping the program name.
        """

        self.arguments = argv[1:]
        self.flag = ""
        self.path = ""

    def parse(self) -> None:
        """
        Parses the command-line arguments to validate a flag and an optional file path.

        The method performs the following checks:
        1. Validates that there is at least one argument (the flag) and no more than two (flag + file path).
        2. Ensures the flag is one of "--training", "--prediction", or "--bonus".
        3. If a file path is provided, it checks:
            - The file has a valid extension (.csv or .json).
            - The file exists on the filesystem.
            - The file is readable.
        If any of these validations fail, a RuntimeError is raised.

        Raises
        ------
        RuntimeError
            If an invalid number of arguments are provided.
            If the flag is invalid.
            If the file path is invalid or inaccessible.
        """

        length: int = len(self.arguments)

        if length < 1 or length > 2:
            raise RuntimeError("One or two arguments are required: a flag and an optional file path.")

        self.flag = self.arguments[0]

        if self.flag not in ["--training", "--prediction", "--bonus"]:
            raise RuntimeError(f"Invalid flag '{self.flag}'. Valid flags are '--training', '--prediction', '--bonus'.")

        if length == 2:
            self.path = self.arguments[1]

            if not (self.path.endswith(".csv") or self.path.endswith(".json")):
                raise RuntimeError(f"Invalid file type '{self.path}'. Only .csv and .json files are allowed.")
            elif not path.isfile(self.path):
                raise RuntimeError(f"File '{self.path}' does not exist.")
            elif not access(self.path, R_OK):
                raise RuntimeError(f"File '{self.path}' is not readable.")
