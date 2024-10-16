#
#   main.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from random import uniform

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

    if parser.flag == "--prediction":
        theta0, theta1, xmin, xmax = file_manager.read_json()
        mileage: float = float(input("Enter the mileage (in km) of the car: "))
        normalized_mileage: float = linear_regression.normalize_input(mileage, xmin, xmax)
        price: float = linear_regression.estimate_price(normalized_mileage, theta0, theta1)
        print(f"The estimated price for a car with {mileage} km is ${price:.2f}")
    else:
        mileages, prices = file_manager.read_csv()

        normalized_mileages: list[list[float]] = linear_regression.normalize(mileages)
        biais: list[list[float]] = [[1.0]] * len(normalized_mileages)

        matrix_x: list[list[float]] = [m + p for m, p in zip(normalized_mileages, biais)]
        matrix_y: list[list[float]] = prices
        theta: list[list[float]] = [[uniform(-0.01, 0.01)], [uniform(-0.01, 0.01)]]

        n_iterations: int = 1000
        learning_rate: float = 0.07

        if parser.flag == "--bonus":
            pass
