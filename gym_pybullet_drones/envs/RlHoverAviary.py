import numpy as np
from dataclasses import dataclass, field
import math
from scipy.optimize import rosen, differential_evolution, dual_annealing

from scipy.optimize import minimize
from scipy.optimize import basinhopping
from scipy.optimize import fmin_l_bfgs_b
from gym_pybullet_drones.envs.NewBaseRLAviary import NewBaseRLAviary
from gym_pybullet_drones.examples.USV_trajectory import UsvTrajectory
from gym_pybullet_drones.examples.gradient_descent import LossFunction
from gym_pybullet_drones.examples.loss_function import LossFunction0
from gym_pybullet_drones.utils.enums import DroneModel, Physics, ActionType, ObservationType
# from mystic.solvers import diffev2, fmin_powell
# from mystic.math import almostEqual

@dataclass(frozen=True)
class TimeData:
    T: float  # длительность во времени
    fs: int  # частота дискретизации
    n: int = field(init=False)  # число отсчетов
    dt: float = field(init=False)  # длительность отсчета времени
    t: np.ndarray = field(init=False)  # отсчеты времени

    def __post_init__(self):
        object.__setattr__(self, 'n', int(self.T * self.fs))
        object.__setattr__(self, 'dt', 1 / self.fs)
        object.__setattr__(self, 't', np.arange(self.n) * self.dt)

    def sample(self, fs):
        return TimeData(T=self.T, fs=fs)


class RlHoverAviary(NewBaseRLAviary):
    """Multi-agent RL problem: leader-follower."""

    ################################################################################

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones: int = 2,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 300,
                 ctrl_freq: int = 60,
                 gui=False,
                 record=False,
                 obs: ObservationType = ObservationType.KIN,
                 act: ActionType = ActionType.PID,
                 traj_uav=None
                 ):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """

        # INIT_XYZS = np.array([
        #     [np.random.uniform(-10.0, 20.0), 10, 10],
        #     [np.random.uniform(-10.0, 20.0), 50, 10]
        # ])
        self.trajs = None
        #self.usv_coord = None
        self.EPISODE_LEN_SEC = 20
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         pyb_freq=pyb_freq,
                         ctrl_freq=ctrl_freq,
                         gui=gui,
                         record=record,
                         obs=obs,
                         act=act

                         )
        self.NUM_USV = 4
        if traj_uav is not None:
            self.trajs = traj_uav
            self.usv_coord = self.trajs.xyz


        self.xyz1 = np.array([[0, 40, 0], [0, 50, 0], [0, 60, 0], [0, 70, 0]])
        self.time_data = TimeData(self.EPISODE_LEN_SEC, pyb_freq)
        #print(self.usv_coord)
        if self.ACT_TYPE == ActionType.VEL or self.ACT_TYPE == ActionType.RPM:
            self.m = 120
        elif self.ACT_TYPE == ActionType.PID:
            self.m = 30

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value.

        Returns
        -------
        float
            The reward.

        """
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])
        uav_coord = np.transpose(np.array([states[:, 0], states[:, 1], states[:, 2]]), (1, 0))
        val = LossFunction0.communication_quality_function(uav_coord, self.usv_coord[self.step_counter, :, :])

        loss_func = lambda x: LossFunction0.communication_quality_function(x.reshape(self.NUM_DRONES, 3),
                                                                            self.usv_coord[self.step_counter, :, :])
        optimized = minimize(loss_func, uav_coord.reshape(6, ))
        opt_x = optimized.x.reshape(self.NUM_DRONES, 3)
        opt_x[:, 2] += 10
        val_opt = LossFunction0.communication_quality_function(opt_x,
                                                              self.usv_coord[self.step_counter, :, :])
        #val_opt = LossFunction0.sum_distant(uav_coord, self.usv_coord[self.step_counter, :, :])
        ret = (val_opt- val) / val_opt
        #ret = 10000 / val**2
        if uav_coord[0, 2] < 1 or uav_coord[1, 2] < 1:
            print("H меньше 1")

        return ret

    ################################################################################

    def step(self, action):

        observation, reward, terminated, truncated, info = super().step(action)
        obs_usv = np.zeros((self.NUM_USV, 6))
        for i in range(self.NUM_USV):
            obs_usv[i, :] = np.hstack([self.usv_coord[self.step_counter-1, i, :],
                                       np.append(self.trajs.v[self.step_counter-1, i, :], [0])]).reshape(6, )

        ret = np.array([obs_usv[i, :] for i in range(self.NUM_USV)]).astype('float32')
        pad_width = ((0, 0), (0, self.m))
        padded_array = np.pad(ret, pad_width, mode='constant', constant_values=0)
        observation = np.concatenate((observation, padded_array), axis=0)
        return observation, reward, terminated, truncated, info

    def reset(self, seed: int = None, options: dict = None):
        if self.trajs is None:
            phi = np.random.uniform(-math.pi, math.pi, self.NUM_USV)
            self.trajs = UsvTrajectory(self.time_data, m=self.NUM_USV, r0=self.xyz1[:, 0:2], xyz0=self.xyz1, φ0=phi)
            self.usv_coord = self.trajs.xyz
        initial_obs, initial_info = super().reset(seed=42, options={})

        obs_usv = np.zeros((self.NUM_USV, 6))
        for i in range(self.NUM_USV):
            obs_usv[i, :] = np.hstack([self.xyz1[i, :],
                                       np.append(np.zeros(2), [0])]).reshape(6, )

        ret = np.array([obs_usv[i, :] for i in range(self.NUM_USV)]).astype('float32')
        pad_width = ((0, 0), (0, self.m))
        padded_array = np.pad(ret, pad_width, mode='constant', constant_values=0)
        initial_obs = np.concatenate((initial_obs, padded_array), axis=0)
        return initial_obs, initial_info

    def _computeTerminated(self):
        """Computes the current done value.

        Returns
        -------
        bool
            Whether the current episode is done.

        """
        return False

    ################################################################################

    def _computeTruncated(self):
        """Computes the current truncated value.

        Returns
        -------
        bool
            Whether the current episode timed out.

        """

        if self.step_counter / self.PYB_FREQ == self.EPISODE_LEN_SEC - 0.1:
            return True
        else:
            return False

    ################################################################################

    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[str, int]
            Dummy value.

        """
        return {"answer": 42}  #### Calculated by the Deep Thought supercomputer in 7.5M years
