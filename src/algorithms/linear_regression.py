#
#   linear_regression.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

class LinearRegression:

    def __init__(self) -> None:
        pass

    def normalize(self, matrix: list[list[float]]) -> list[list[float]]:
        """
        Normalizes the values in a 2D matrix (list of lists).

        This method takes a 2D matrix of floating-point numbers and normalizes each value based on the formula:
        `(value - min) / (max - min)`, where `min` is the smallest value in the matrix and `max` is the largest value in
        the matrix. The normalization ensures that all values in the matrix are scaled between 0 and 1, based on the
        minimum and maximum values.

        It utilizes the private methods `__matrix_min` and `__matrix_max` to find the smallest and largest values in the
        matrix.

        Parameters
        ----------
        matrix : list[list[float]]
            A 2D list (matrix) where each inner list represents a row containing floating-point numbers.

        Returns
        -------
        list[list[float]]
            A new 2D list where each value in the original matrix has been normalized between 0 and 1.

        Example
        -------
        matrix = [[1.0], [5.0], [10.0]]
        result = normalize(matrix)  # Returns [[0.0], [0.4444], [1.0]]

        Notes
        -----
        - If all the values in the matrix are the same, the normalization formula will result in division by zero.
          Handling such edge cases should be added based on your needs.
        """

        xmin: float = self.__matrix_min(matrix)
        xmax: float = self.__matrix_max(matrix)
        return [[(item[0] - xmin) / (xmax - xmin)] for item in matrix]

    def normalize_input(self, mileage: float, xmin: float, xmax: float) -> float:
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

    ## MARK: - Private Methods

    def __matrix_min(self, matrix: list[list[float]]) -> float:
        """
        Finds the minimum value in a 2D matrix (list of lists).

        This private method iterates through each row of the matrix and computes the minimum value for each row. It then
        returns the smallest of these values, effectively providing the minimum value of the entire matrix.

        Parameters
        ----------
        matrix : list[list[float]]
            A 2D list (matrix) where each inner list represents a row containing floating-point numbers.

        Returns
        -------
        float
            The minimum value found in the matrix.

        Example
        -------
        matrix = [[1.0, 2.5, 3.2], [4.0, 0.1, 5.7]]
        result = __matrix_min(matrix)  # Returns 0.1
        """

        return min(min(row) for row in matrix)

    def __matrix_max(self, matrix: list[list[float]]) -> float:
        """
        Finds the maximum value in a 2D matrix (list of lists).

        This private method iterates through each row of the matrix and computes the maximum value for each row. It then
        returns the largest of these values, effectively providing the maximum value of the entire matrix.

        Parameters
        ----------
        matrix : list[list[float]]
            A 2D list (matrix) where each inner list represents a row containing floating-point numbers.

        Returns
        -------
        float
            The maximum value found in the matrix.

        Example
        -------
        matrix = [[1.0, 2.5, 3.2], [4.0, 0.1, 5.7]]
        result = __matrix_max(matrix)  # Returns 5.7
        """

        return max(max(row) for row in matrix)

    def __matrix_multiply(self, a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        """
        Multiplies two matrices and returns the resulting matrix.

        Parameters
        ----------
        a : list[list[float]]
            The first matrix to multiply, with dimensions m x n.
        b : list[list[float]]
            The second matrix to multiply, with dimensions n x p.

        Returns
        -------
        list[list[float]]
            The resulting matrix of dimensions m x p.

        Raises
        ------
        ValueError
            If the number of columns in the first matrix does not match the number of rows in the second matrix.

        Notes
        -----
        - Matrix multiplication is only possible if the number of columns in matrix `a` matches the number of rows in
          matrix `b`.
        - The returned matrix will have dimensions of the number of rows in `a` and the number of columns in `b`.
        """

        if len(a[0]) != len(b):
            raise ValueError("The number of columns in the first matrix must correspond to the number of rows in the second matrix.")
        transposed: list[tuple[float]] = list(zip(*b))
        return [[sum(x * y for x, y in zip(row_a, col_b)) for col_b in transposed] for row_a in a]

    def __matrix_substract(self, a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        """
        Subtracts one matrix from another element-wise.

        Parameters
        ----------
        a : list[list[float]]
            The first matrix (minuend).
        b : list[list[float]]
            The second matrix (subtrahend).

        Returns
        -------
        list[list[float]]
            The resulting matrix after subtraction.

        Notes
        -----
        - The matrices `a` and `b` must have the same dimensions.
        - Subtracts each element of `b` from the corresponding element in `a`.
        """

        return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

    def __matrix_square(self, matrix: list[list[float]]) -> list[list[float]]:
        """
        Squares each element of a matrix.

        Parameters
        ----------
        matrix : list[list[float]]
            The matrix to square element-wise.

        Returns
        -------
        list[list[float]]
            The resulting matrix where each element is squared.

        Notes
        -----
        - Each element in the matrix is raised to the power of 2.
        """

        return [[matrix[i][j] ** 2 for j in range(len(matrix[0]))] for i in range(len(matrix))]

    def __matrix_sum(self, matrix: list[list[float]]) -> float:
        """
        Computes the sum of all elements in a matrix.

        Parameters
        ----------
        matrix : list[list[float]]
            The matrix whose elements will be summed.

        Returns
        -------
        float
            The sum of all the elements in the matrix.

        Notes
        -----
        - Iterates over all rows and columns to compute the sum.
        """

        return sum(sum(row) for row in matrix)

    def __matrix_transpose(self, matrix: list[list[float]]) -> list[float]:
        """
        Transposes a matrix (switches rows with columns).

        Parameters
        ----------
        matrix : list[list[float]]
            The matrix to transpose.

        Returns
        -------
        list[list[float]]
            The transposed matrix.

        Notes
        -----
        - The resulting matrix will have its rows and columns swapped.
        """

        return list(map(list, zip(*matrix)))

    def __matrix_scalar(self, matrix: list[list[float]], scalar: float) -> list[list[float]]:
        """
        Multiplies each element of a matrix by a scalar.

        Parameters
        ----------
        matrix : list[list[float]]
            The matrix to multiply by the scalar.
        scalar : float
            The scalar value to multiply each matrix element by.

        Returns
        -------
        list[list[float]]
            The resulting matrix after scalar multiplication.

        Notes
        -----
        - Each element in the matrix is multiplied by the scalar.
        """

        return [[element * scalar for element in row] for row in matrix]

    def __matrix_mean(self, matrix: list[list[float]]) -> float:
        """
        Computes the mean (average) of all elements in a matrix.

        Parameters
        ----------
        matrix : list[list[float]]
            The matrix whose mean will be computed.

        Returns
        -------
        float
            The mean value of all elements in the matrix.

        Notes
        -----
        - The mean is calculated by dividing the sum of all elements by the total number of elements.
        - Uses `__matrix_sum` to calculate the sum of all elements.
        """

        return self.__matrix_sum(matrix) / (len(matrix) * len(matrix[0]))
