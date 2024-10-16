#
#   file_manager.py
#   Linear-Regression
#
#   Created by Marc MOSCA on 15/10/2024.
#

from csv import reader
from json import dump, load, JSONDecodeError
from typing import Any


class FileManager:
    """
    A class to manage reading from and writing to files. This class supports reading from CSV and JSON files, as well as
    writing to JSON files.

    Attributes
    ----------
    path : str
        The file path used by the instance for reading operations (either CSV or JSON).
    """

    path: str

    def __init__(self, path: str) -> None:
        """
        Initializes the FileManager instance with a file path for subsequent read operations.

        Parameters
        ----------
        path : str
            The path to the file that will be read (CSV or JSON).
        """

        self.path = path

    def read_csv(self) -> tuple[list[list[float]], list[list[float]]]:
        """
        Reads a CSV file and extracts mileage and price data.

        This method reads a CSV file where the first column contains mileage values and the second column contains price
        values. The mileage and price values are extracted, converted to floating-point numbers, and stored in two
        separate 2D lists. Each value is encapsulated in its own list, effectively creating a column vector format for
        further matrix-based operations.

        Returns
        -------
        tuple[list[list[float]], list[list[float]]]
            A tuple containing two 2D lists:
            - The first list represents mileage values, where each element is a list containing a single float.
            - The second list represents price values, where each element is a list containing a single float.

        Raises
        ------
        RuntimeError
            If the file does not exist or an error occurs during reading.

        Example
        -------
        If the CSV file contains:

        ```
        km,price
        240000,3650
        139800,3800
        ```

        After calling `read_csv()`, the result will be:
        ([[240000.0], [139800.0]], [[3650.0], [3800.0]])

        Notes
        -----
        - The CSV file must have a header row, which will be skipped.
        - Each mileage and price value is stored as a float in a list for compatibility with matrix-based operations.
        """

        mileage: list[list[float]] = []
        price: list[list[float]] = []

        try:
            with open(self.path, mode='r', newline='') as file:
                r = reader(file)
                next(r)
                for row in r:
                    mileage.append([float(row[0])])
                    price.append([float(row[1])])

            return mileage, price
        except FileNotFoundError:
            raise RuntimeError(f"The file {self.path} does not exist.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while reading the CSV file: {e}")

    def read_json(self) -> tuple[float, float, float, float]:
        """
        Reads a JSON file and extracts the model parameters for linear regression.

        This method reads a JSON file from the specified path and extracts four key parameters:
        `theta0`, `theta1`, `xmin`, and `xmax`, which are used in the linear regression model.

        In case of a JSON decoding error, it raises a `RuntimeError` to indicate invalid JSON format.
        If any other exception occurs (such as file-related issues), it returns default values of
        0.0 for `theta0`, `theta1`, and `xmin`, and 1.0 for `xmax` to ensure the program doesn't crash.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple containing four values:
            - theta0 (float): The intercept of the linear regression model.
            - theta1 (float): The slope of the linear regression model.
            - xmin (float): The minimum value for mileage used in normalization.
            - xmax (float): The maximum value for mileage used in normalization.

        Raises
        ------
        RuntimeError
            If the JSON file contains invalid JSON data.
        """

        try:
            with open(self.path, mode='r') as file:
                data: dict[str, Any] = load(file)
            return data['theta0'], data['theta1'], data['xmin'], data['xmax']
        except JSONDecodeError:
            raise RuntimeError(f"The file {self.path} contains invalid JSON.")
        except Exception as e:
            return 0.0, 0.0, 0.0, 1.0

    def write_json(self, path: str, data: dict[str, Any]) -> None:
        """
        Writes the given data into a JSON file at the specified path.

        The file will be created or overwritten if it already exists, and the data will be formatted with an indentation
        of 4 spaces for readability.

        Parameters
        ----------
        path : str
            The path where the JSON file will be written.
        data : dict[str, Any]
            The data to write into the JSON file.

        Raises
        ------
        RuntimeError
            If an error occurs during the writing process.
        """

        try:
            with open(path, mode='w') as file:
                dump(data, file, indent=4)
        except Exception as e:
            raise RuntimeError(f"An error occurred while writing to the JSON file: {e}")
