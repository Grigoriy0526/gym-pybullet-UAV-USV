import math

import numpy as np


class LossFunction0:
    def __init__(self, x, usv_coord):
        self.x = x
        self.USV_coord = usv_coord

    @staticmethod
    def communication_quality_function(x, usv_coord):
        norm = np.linalg.norm(x[:, None] - x[None], axis=-1) ** 2
        uav_sum_dist = np.sum(norm) / 2
        alpha = math.pi / 10
        d = x[:, None] - usv_coord[None]
        d_norm = np.linalg.norm(d, axis=-1)
        angle = np.arctan(d[:, :, 2] / np.linalg.norm(d[:, :, :2], axis=-1))
        k = (1 - 100) * angle / alpha + 100
        new_d_norm = np.where(angle < alpha, d_norm * k, d_norm)
        uav_usv_sum_dist = np.sum(np.min(new_d_norm, axis=0) ** 2, axis=0)

        return uav_usv_sum_dist + 0.5 * uav_sum_dist