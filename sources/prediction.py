#
#   prediction.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from linear_regression import model, normalize
from reader import reader_json


(theta, xmin, xmax) = reader_json("./models/trained_model.json")
mileage: float = float(input("Enter the mileage (in km) of the car: "))
normalized_mileage: float = normalize(mileage, xmin, xmax)
price: float = model(normalized_mileage, theta)
print(f"The estimated price for a car with {mileage} km is ${price:.2f}")
