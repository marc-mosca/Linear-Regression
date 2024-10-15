#
#   main.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from managers.file_manager import FileManager
from managers.parser_manager import ParserManager


if __name__ == '__main__':
    parser: ParserManager = ParserManager()
    parser.parse()
    print(f"Flag: '{parser.flag}'\nPath: '{parser.path}'")

    file_manager: FileManager

    if len(parser.path) != 0:
        file_manager = FileManager(parser.path)
    else:
        file_manager = FileManager("./assets/data/car-price.csv")

    data = file_manager.read_csv()
    print(data)
