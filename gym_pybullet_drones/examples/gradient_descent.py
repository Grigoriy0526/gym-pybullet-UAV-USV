import math

import numpy as np


def BellmanFord(graph, source):
    distance = [float("Inf")] * len(graph)
    distance[source] = 0

    for _ in range(len(graph) - 1):
        for u, v, w in graph:
            if distance[u] != float("Inf") and distance[u] + w < distance[v]:
                distance[v] = distance[u] + w

    for u, v, w in graph:
        if distance[u] != float("Inf") and distance[u] + w < distance[v]:
            return

    return distance


class LossFunction:
    def __init__(self, x, usv_coord):
        self.x = x
        self.USV_coord = usv_coord

    @staticmethod
    def communication_quality_function(x, usv_coord):
        # norm = np.linalg.norm(x[:, :, None] - x[:, None], axis=-1) ** 2
        # uav_sum_dist = np.sum(norm.reshape(norm.shape[0], -1), axis=1) / 2
        # uav_usv_sum_dist = np.sum(np.min(np.linalg.norm(x[:, :, None] - usv_coord[:, None], axis=-1), axis=1) ** 2,
        #                           axis=1)
        # return uav_usv_sum_dist + 0.5 * uav_sum_dist
        alpha = math.pi / 10
        d = x[:, None] - usv_coord[None]
        angle = np.arctan(d[:, :, 2] / np.linalg.norm(d[:, :, :2], axis=-1))
        k = (1-100)*angle/alpha + 100
        norm_usv_uav = np.where(angle < alpha, (np.linalg.norm(d, axis=-1) ** 2) * k, np.linalg.norm(d, axis=-1) ** 2)

        norm_uav = np.linalg.norm(x[:, None] - x[None], axis=-1) ** 2

        matrix = np.block([[np.ones((4, 4), int) * np.inf, norm_usv_uav.T],
                            [norm_usv_uav, norm_uav]
                           ])

        ind = np.indices((6, 6)).reshape(2, -1).T
        y = [[i[0], i[1], matrix[tuple(i)]] for i in ind if matrix[tuple(i)] < np.inf]

        # graph = [(0, 4, norm_usv_uav[0, 0, 0]),
        #          (0, 5, norm_usv_uav[0, 1, 0]),
        #          (1, 4, norm_usv_uav[0, 0, 1]),
        #          (1, 5, norm_usv_uav[0, 1, 1]),
        #          (2, 4, norm_usv_uav[0, 0, 2]),
        #          (2, 5, norm_usv_uav[0, 1, 2]),
        #          (3, 4, norm_usv_uav[0, 0, 3]),
        #          (3, 5, norm_usv_uav[0, 1, 3]),
        #          (4, 5, norm_uav[0, 0, 1]),
        #          (4, 0, norm_usv_uav[0, 0, 0]),
        #          (4, 1, norm_usv_uav[0, 0, 1]),
        #          (4, 2, norm_usv_uav[0, 0, 2]),
        #          (4, 3, norm_usv_uav[0, 0, 3]),
        #          (5, 0, norm_usv_uav[0, 1, 0]),
        #          (5, 1, norm_usv_uav[0, 1, 1]),
        #          (5, 2, norm_usv_uav[0, 1, 2]),
        #          (5, 3, norm_usv_uav[0, 1, 3])
        #         ]

        S = np.zeros(4)

        for i in range(4):
            shortest_distances = BellmanFord(y, i)
            S[i] = np.sum(shortest_distances[0:4])

        return np.sum(S) / 2
