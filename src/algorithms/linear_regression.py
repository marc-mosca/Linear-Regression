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
