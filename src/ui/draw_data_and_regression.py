#
#   draw_data_and_regression.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 16/10/2024.
#

import matplotlib.pyplot as plt


def draw_data_and_regression(mileages: list[list[float]], prices: list[list[float]], theta: list[list[float]], xmin: float, xmax: float) -> None:
    """
    Plots the raw data points (mileages and prices) and the regression line based on the learned parameters (theta).

    Parameters
    ----------
    mileages : list[list[float]]
        A matrix where each element is a list containing the mileage values of cars.
    prices : list[list[float]]
        A matrix where each element is a list containing the price values of cars.
    theta : list[list[float]]
        The parameter matrix (weights) of the linear regression model, where theta[0][0] is the intercept (theta0) and
        theta[1][0] is the slope (theta1).
    xmin : float
        The minimum value of the mileage used for normalization.
    xmax : float
        The maximum value of the mileage used for normalization.

    Returns
    -------
    None
        This function does not return any value. It plots the scatter plot of the data points and the regression line.

    Notes
    -----
    - The function normalizes the mileage range from `xmin` to `xmax` and calculates predicted prices using the
      regression model.
    - The scatter plot displays the original mileage and price data, while the red line shows the predicted price using
      the regression model.
    - The `theta` matrix should be a 2x1 matrix representing the parameters of the linear regression (theta0, theta1).
    """

    mileage: list[float] = [m[0] for m in mileages]
    price: list[float] = [p[0] for p in prices]
    theta0, theta1 = theta[0][0], theta[1][0]
    iterations: int = 100
    mileages_range: list[float] = [xmin + i * ((xmax - xmin) / (iterations - 1)) for i in range(iterations)]
    prices_predictions: list[float] = [theta1 + theta0 * (km_value - xmin) / (xmax - xmin) for km_value in mileages_range]

    plt.scatter(mileage, price, color="blue", label="Data Points")
    plt.plot(mileages_range, prices_predictions, color="red", label="Regression Line")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price ($)")
    plt.title("Car Price Prediction vs Mileage")
    plt.legend()
    plt.show()
