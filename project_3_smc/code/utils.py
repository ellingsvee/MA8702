from os import PathLike
from pathlib import Path
import numpy as np


def load_sensor_data(folder_path: PathLike = Path("sensor_data")) -> np.ndarray:
    """Load the sensor data from the given folder path.

    Args:
        folder_path (PathLike, optional): The path to the folder containing the sensor data. Defaults to Path("sensor_data").

    Returns:
        np.ndarray: The loaded sensor data as a NumPy array.
    """
    data = []
    for file in Path(folder_path).glob("*.txt"):
        data.append(np.loadtxt(file))
    # return np.concatenate(data, axis=0)
    return np.stack(data, axis=0)
