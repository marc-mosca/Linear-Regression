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

try:
    mileage: str = input("Enter the mileage (in km) of the car: ")

    if not mileage.isdigit():
        exit(f"Mileage must be a positive number: {mileage}")

    mileage: float = float(mileage)

    normalized_mileage: float = normalize(mileage, xmin, xmax)
    price: float = model(normalized_mileage, theta)

    if price < 0:
        exit("The mileage is too high to set a price greater than 0.")

    print(f"The estimated price for a car with {mileage} km is ${price:.2f}")
except KeyboardInterrupt:
    exit("\nError: handle keyboard interruption (ctrl-c).")
except EOFError:
    exit("\nError: handle end of line (ctrl-d).")
