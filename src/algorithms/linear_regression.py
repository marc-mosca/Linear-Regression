#
#   linear_regression.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

class LinearRegression:

    def __init__(self) -> None:
        pass

    def normalize(self, x: list[int]) -> tuple[list[float], int, int]:
        """
        Normalizes a list of integer values to the range [0, 1].

        This method takes a list of integers as input, calculates the minimum and maximum values, and returns the
        normalized values as a list of floats.
        The normalized values are scaled using the formula:

            normalized_value = (value - min_value) / (max_value - min_value)

        The method rounds each normalized value to 8 decimal places.

        Parameters
        ----------
        x : list[int]
            A list of integers representing the values to be normalized.

        Returns
        -------
        tuple[list[float], int, int]
            A tuple containing:
            - A list of normalized values as floats.
            - The minimum value from the input list.
            - The maximum value from the input list.

        Example
        -------
        >>> model = LinearRegression()
        >>> x = [100, 200, 300]
        >>> normalized, xmin, xmax = model.normalize(x)
        >>> print(normalized)
        [0.0, 0.5, 1.0]
        >>> print(xmin, xmax)
        100 300
        """

        xmin: int = min(x)
        xmax: int = max(x)
        nx: list[float] = [round((y - xmin) / (xmax - xmin), 8) for y in x]
        return nx, xmin, xmax

    def normalize_input(self, mileage: float, xmin: int, xmax: int) -> float:
        """
        Normalizes the input mileage based on the minimum and maximum mileage values.

        This method scales the mileage to a range between 0 and 1 using the provided minimum (xmin) and maximum (xmax)
        mileage values. Normalization helps to bring the mileage into a consistent range for model estimation.

        Parameters
        ----------
        mileage : float
            The mileage of the car to be normalized.

        xmin : int
            The minimum mileage value used for normalization.

        xmax : int
            The maximum mileage value used for normalization.

        Returns
        -------
        float
            The normalized mileage value, calculated as (mileage - xmin) / (xmax - xmin).
        """

        return (mileage - xmin) / (xmax - xmin)

    def estimate_price(self, mileage: float, theta0: float, theta1: float) -> float:
        """
        Estimates the price of a car based on its normalized mileage using a linear model.

        This method uses the linear regression formula `price = theta0 + theta1 * mileage` to estimate the price of the
        car. The mileage is expected to be normalized (scaled between 0 and 1).

        Parameters
        ----------
        mileage : float
            The normalized mileage of the car (between 0 and 1).

        theta0 : float
            The intercept of the linear regression model.

        theta1 : float
            The slope of the linear regression model.

        Returns
        -------
        float
            The estimated price of the car, based on the linear model.
        """

        return theta0 + (theta1 * mileage)
