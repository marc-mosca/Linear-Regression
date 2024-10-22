#
#   training.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from linear_regression import gradient_descent, normalize
from matrix import Matrix
from printer import print_dataset, print_normalized
from reader import reader_csv
from writer import writer_json


iterations: int = 1_000
learning_rate: float = 0.07

(target, features) = reader_csv("./data/dataset.csv")
print_dataset(target, features)

xmin: float = features.min()
xmax: float = features.max()

normalized_features: Matrix = Matrix([[normalize(value, xmin, xmax)] for [value] in features.get()])
bias: Matrix = Matrix([[1.0]] * target.shape()[0])

print_normalized(features, normalized_features)

x: Matrix = Matrix([f + b for f, b in zip(normalized_features.get(), bias.get())])
y: Matrix = target
theta: Matrix = Matrix([[0.0], [0.0]])

theta = gradient_descent(x, y, theta, learning_rate, iterations)

data: dict[str: float] = {"theta0": theta.get()[0][0], "theta1": theta.get()[1][0], "xmin": xmin, "xmax": xmax}
writer_json("./models/trained_model.json", data)

print(f"Training completed. Model saved in './models/trained_model.json'.")
