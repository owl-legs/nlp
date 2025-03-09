from typing import Callable, Union
import numpy as np
from embeddings.analysis.distance_functions import euclidean_distance


def distance_matrix(coordinates: list[list[Union[float, int]]], distance: Callable = euclidean_distance) -> list[list[float]]:

    n = len(coordinates)
    _distance_matrix = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(i):

            _distance = distance(x=coordinates[i], y=coordinates[j])
            _distance_matrix[i][j] = _distance
            _distance_matrix[j][i] = _distance

    return _distance_matrix.tolist()


def nearest_neighbors(
        coordinates: list[list[Union[float, int]]],
        distance: Callable,
        n_neighbors: int = 10
) -> list[list[int]]:

    n = len(coordinates)
    _nearest_neighbors = []

    _distance_matrix = distance_matrix(
        coordinates=coordinates,
        distance=distance
    )

    for i in range(n):
        _distances = _distance_matrix[i]
        _distances = _distances.sort()
        _nearest_neighbors.append(_distances[:n_neighbors])

    return _nearest_neighbors




