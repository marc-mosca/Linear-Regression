#
#   draw_raw_mileage.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

import matplotlib.pyplot as plt


def draw_raw_mileage(mileage: list[int], price: list[int]) -> None:
    """
    Plots the relationship between mileage and price using a scatter plot.

    This function takes a list of mileage values and their corresponding price values, and creates a scatter plot to
    visualize the relationship between the mileage (in kilometers) and the price (in dollars). It helps to analyze how
    price changes based on mileage.

    Parameters
    ----------
    mileage : list[int]
        A list of integer values representing the mileage of vehicles (in kilometers).

    price : list[int]
        A list of integer values representing the price of vehicles (in dollars).

    Example
    -------
    >>> mileage = [240000, 139800, 150500, 185530]
    >>> price = [3650, 3800, 4400, 4450]
    >>> draw_raw_mileage(mileage, price)

    This will generate a scatter plot showing the price of vehicles as a function of mileage.
    """

    plt.scatter(mileage, price, color='blue', label='Données')

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price ($)")
    plt.title("Price per kilometer (raw data)")
    plt.legend()
    plt.show()
