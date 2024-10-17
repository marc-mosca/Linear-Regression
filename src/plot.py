#
#   plot.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

import matplotlib.pyplot as plt
from matrix import Matrix


class Plot(object):
    def __init__(self):
        pass

    def draw_dataset(self, target: Matrix, features: Matrix) -> None:
        mileage: list[float] = [m[0] for m in features.values]
        price: list[float] = [p[0] for p in target.values]
        plt.scatter(mileage, price, color='blue', label='Données')
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price ($)")
        plt.title("Price per kilometer (raw data)")
        plt.legend()
        plt.show()

    def draw_normalize_features(self, features: Matrix, normalized_features: Matrix) -> None:
        mileage: list[float] = [m[0] for m in features.values]
        normalized_mileage: list[float] = [p[0] for p in normalized_features.values]
        plt.plot(mileage, normalized_mileage, 'bo-', label="Normalized Mileage")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Normalized Mileage")
        plt.title("Normalized mileage vs. gross mileage")
        plt.legend()
        plt.grid(True)
        plt.show()

    def draw_regression(self, target: Matrix, features: Matrix, theta: Matrix, xmin: float, xmax: float) -> None:
        mileage: list[float] = [m[0] for m in features.values]
        price: list[float] = [p[0] for p in target.values]
        theta0, theta1 = theta.values[0][0], theta.values[1][0]
        iterations: int = 100
        mileages_range: list[float] = [xmin + i * ((xmax - xmin) / (iterations - 1)) for i in range(iterations)]
        prices_predictions: list[float] = [theta1 + theta0 * (km - xmin) / (xmax - xmin) for km in mileages_range]
        plt.scatter(mileage, price, color="blue", label="Data Points")
        plt.plot(mileages_range, prices_predictions, color="red", label="Regression Line")
        plt.xlabel("Mileage (km)")
        plt.ylabel("Price ($)")
        plt.title("Car Price Prediction vs Mileage")
        plt.legend()
        plt.show()
