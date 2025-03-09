from typing import Union

import numpy as np


def euclidean_distance(x: list[Union[float, int]], y: list[Union[float, int]]) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sqrt(np.sum((x - y)**2))
