#
#   printer.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from matrix import Matrix
from matplotlib.pyplot import grid, legend, plot, scatter, show, title, xlabel, ylabel


def print_matrix(title: str, matrix: Matrix) -> None:
    print(f"{title}:\n{matrix}")

def print_cost(cost_history: list[float]) -> None:
    plot(cost_history)
    xlabel('Itération')
    ylabel('Coût')
    title('Évolution du coût')
    show()

def print_dataset(target: Matrix, features: Matrix) -> None:
    mileage: list[float] = [m[0] for m in features.get()]
    price: list[float] = [p[0] for p in target.get()]
    scatter(mileage, price, color='blue', label='Données')
    xlabel("Mileage (km)")
    ylabel("Price ($)")
    title("Price per kilometer (raw data)")
    legend()
    show()

def print_normalized(features: Matrix, normalized_features: Matrix) -> None:
    mileage: list[float] = [m[0] for m in features.get()]
    normalized_mileage: list[float] = [p[0] for p in normalized_features.get()]
    plot(mileage, normalized_mileage, 'bo-', label="Normalized Mileage")
    xlabel("Mileage (km)")
    ylabel("Normalized Mileage")
    title("Normalized mileage vs. gross mileage")
    legend()
    grid(True)
    show()

def print_regression(target: Matrix, features: Matrix, theta: Matrix, xmin: float, xmax: float) -> None:
    mileage: list[float] = [m[0] for m in features.get()]
    price: list[float] = [p[0] for p in target.get()]
    theta_values = theta.get()
    theta0, theta1 = theta_values[0][0], theta_values[1][0]
    iterations: int = 100
    mileages_range: list[float] = [xmin + i * ((xmax - xmin) / (iterations - 1)) for i in range(iterations)]
    prices_predictions: list[float] = [theta1 + theta0 * (km - xmin) / (xmax - xmin) for km in mileages_range]
    scatter(mileage, price, color="blue", label="Data Points")
    plot(mileages_range, prices_predictions, color="red", label="Regression Line")
    xlabel("Mileage (km)")
    ylabel("Price ($)")
    title("Car Price Prediction vs Mileage")
    legend()
    show()
