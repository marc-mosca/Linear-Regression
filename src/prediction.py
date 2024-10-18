#
#   prediction.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from config import TRAINED_MODEL_PATH
from linear_regression import LinearRegression
from utils import read_json


if __name__ == '__main__':
    linear_regression: LinearRegression = LinearRegression(1000, 0.07)

    theta, xmin, xmax = read_json(TRAINED_MODEL_PATH)
    mileage: float = float(input("Enter the mileage (in km) of the car: "))
    normalized_mileage: float = linear_regression.normalize(mileage, xmin, xmax)
    price: float = linear_regression.model(normalized_mileage, theta)
    print(f"The estimated price for a car with {mileage} km is ${price:.2f}")
