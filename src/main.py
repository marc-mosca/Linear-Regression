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
        path = parser.path if len(parser.path) > 0 else "./assets/data/data.csv"
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
        learning_rate: float = 0.12

        theta = linear_regression.gradient_descent(matrix_x, matrix_y, theta, learning_rate, n_iterations)

        data: dict[str: float] = {
            "theta0": theta[1][0],
            "theta1": theta[0][0],
            "xmin": min(min(row) for row in mileages),
            "xmax": max(max(row) for row in mileages)
        }
        file_manager.write_json("./assets/data/linear_regression.json", data)

        print(f"Training completed. Model saved in ./assets/data/linear_regression.json")

        if parser.flag == "--bonus":
            predictions: list[list[float]] = linear_regression.model(matrix_x, theta)
            r_squared: float = linear_regression.coefficients_determination(matrix_y, predictions)
            print(f"Model accuracy (R^2 score): {round(r_squared, 4) * 100} %")
