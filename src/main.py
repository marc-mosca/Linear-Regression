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

    path: str = ""

    if parser.flag == "--training" or parser.flag == "--bonus":
        path = parser.path if len(parser.path) > 0 else "./assets/data/car-price.csv"
    else:
        path = parser.path if len(parser.path) > 0 else "./assets/data/linear_regression.json"

    file_manager: FileManager = FileManager(path)

    if parser.flag == "--training":
        km, price = file_manager.read_csv()
        print(f"Km: {km}\nPrice: {price}")
    elif parser.flag == "--prediction":
        pass
    elif parser.flag == "--bonus":
        pass
