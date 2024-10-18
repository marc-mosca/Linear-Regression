#
#   training.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from linear_regression import LinearRegression
from plot import Plot
from config import TRAINED_MODEL_PATH, DATASET_PATH
from dataset import Dataset
from matrix import Matrix
from utils import write_json


if __name__ == '__main__':
    plot: Plot = Plot()
    linear_regression: LinearRegression = LinearRegression(1000, 0.07)

    dataset: Dataset = Dataset(DATASET_PATH)
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
