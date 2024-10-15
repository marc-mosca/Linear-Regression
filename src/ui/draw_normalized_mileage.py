#
#   draw_normalized_mileage.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

import matplotlib.pyplot as plt


def draw_normalized_mileage(mileage: list[int], normalized_mileage: list[float]) -> None:
    """
    Plots the normalized mileage against the original mileage using a line plot.

    This function takes a list of original mileage values and their corresponding normalized mileage values, and plots
    them using Matplotlib. The plot shows the relationship between the gross mileage (in kilometers) and the normalized
    mileage, providing a visual representation of the data normalization.

    Parameters
    ----------
    mileage : list[int]
        A list of integer values representing the gross mileage (in kilometers).

    normalized_mileage : list[float]
        A list of float values representing the normalized mileage, scaled between 0 and 1.

    Example
    -------
    >>> mileage = [240000, 139800, 150500, 185530]
    >>> normalized_mileage = [1.0, 0.538, 0.587, 0.749]
    >>> draw_normalized_mileage(mileage, normalized_mileage)

    This will generate a plot showing the normalized mileage in relation to the gross mileage.
    """

    plt.plot(mileage, normalized_mileage, 'bo-', label="Normalized Mileage")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Normalized Mileage")
    plt.title("Normalized mileage vs. gross mileage")
    plt.legend()
    plt.grid(True)
    plt.show()
