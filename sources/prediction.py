#
#   prediction.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from linear_regression import model, normalize
from reader import reader_json
from sys import exit


(theta, xmin, xmax) = reader_json("./models/trained_model.json")
mileage: float = float(input("Enter the mileage (in km) of the car: "))

if mileage <= 0.0:
    exit(f"Mileage cannot be equal or less than zero: {mileage}")

normalized_mileage: float = normalize(mileage, xmin, xmax)
price: float = model(normalized_mileage, theta)

if price < 0:
    exit("The mileage is too high to set a price greater than 0.")

print(f"The estimated price for a car with {mileage} km is ${price:.2f}")
