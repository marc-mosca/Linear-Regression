#
#   main.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from config import TRAINED_MODEL_PATH
from dataset import Dataset
from linear_regression import LinearRegression
from matrix import Matrix
from parser_manager import ParserManager
from plot import Plot
from utils import read_json, write_json


if __name__ == '__main__':
    parser: ParserManager = ParserManager()
    parser.parse()

    plot: Plot = Plot()
    linear_regression: LinearRegression = LinearRegression(1000, 0.07)

    if parser.flag == "--prediction":
        theta, xmin, xmax = read_json(parser.path)
        mileage: float = float(input("Enter the mileage (in km) of the car: "))
        normalized_mileage: float = linear_regression.normalize(mileage, xmin, xmax)
        price: float = linear_regression.model(normalized_mileage, theta)
        print(f"The estimated price for a car with {mileage} km is ${price:.2f}")
    else:
        dataset: Dataset = Dataset(parser.path)
        plot.draw_dataset(dataset.target, dataset.features)

        xmin: float = dataset.features.min()
        xmax: float = dataset.features.max()

        normalized_features: Matrix = Matrix([[linear_regression.normalize(value, xmin, xmax)] for [value] in dataset.features.values])
        bias: Matrix = Matrix([[1.0]] * len(dataset.target.values))

        plot.draw_normalize_features(dataset.features, normalized_features)

        x: Matrix = Matrix([f + b for f, b in zip(normalized_features.values, bias.values)])
        y: Matrix = dataset.target
        theta: Matrix = Matrix([[0.0], [0.0]])

        theta = linear_regression.gradient_descent(x, y, theta)

        data: dict[str: float] = {"theta0": theta.values[0][0], "theta1": theta.values[1][0], "xmin": xmin, "xmax": xmax}
        write_json(TRAINED_MODEL_PATH, data)

        print(f"Training completed. Model saved in '{TRAINED_MODEL_PATH}'.")

        if parser.flag == "--bonus":
            plot.draw_regression(dataset.target, dataset.features, theta, xmin, xmax)
            predictions: Matrix = linear_regression.matrix_model(x, theta)
            r_squared: float = linear_regression.coefficient_determination(y, predictions)
            print(f"Model accuracy (R^2 score): {r_squared:.4f}.")
