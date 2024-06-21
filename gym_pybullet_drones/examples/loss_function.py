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
        # angle = np.arctan(d[:, :, 2] / np.linalg.norm(d[:, :, :2], axis=-1))
        # k = (1 - 100) * angle / alpha + 100
        # new_d_norm = np.where(angle < alpha, d_norm * k, d_norm)

        # cf = 3/7
        # h_min = (d_norm * np.sqrt(np.clip(1 - cf / d_norm + cf ** 2 / 4, 1, 10000)) - d_norm) / (cf / 2)
        # h = x[:, 2]
        # h_real = np.repeat(h[:, np.newaxis], 4, axis=1)
        # k = (1 - 100) * h_min / h_real + 100
        #new_d_norm = np.where(h_real < h_min, d_norm * k, d_norm)
        uav_usv_sum_dist = np.sum(np.min(d_norm, axis=0) ** 2, axis=0)

        return uav_usv_sum_dist + 0.5 * uav_sum_dist

    @staticmethod
    def sum_distant(x, usv_coord):
        # norm = np.linalg.norm(x[:, None] - x[None], axis=-1) ** 2
        # uav_sum_dist = np.sum(norm) / 2
        # d = x[:, None] - usv_coord[None]
        # d_norm = np.linalg.norm(d, axis=-1)
        # uav_usv_sum_dist = np.sum(d_norm**2)
        norm_usv = np.linalg.norm(usv_coord[:, None] - usv_coord[None], axis=-1) ** 2 / 4
        sum_usv = np.sum(norm_usv**2)
        return sum_usv
