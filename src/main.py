#
#   main.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from algorithms.linear_regression import LinearRegression

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
    linear_regression: LinearRegression = LinearRegression()

    if parser.flag == "--training":
        mileage, price = file_manager.read_csv()
        normalized_mileage, xmin, xmax = linear_regression.normalize(mileage)
    elif parser.flag == "--prediction":
        pass
    elif parser.flag == "--bonus":
        pass
