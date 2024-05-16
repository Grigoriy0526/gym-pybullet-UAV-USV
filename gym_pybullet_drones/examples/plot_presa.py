import os
from datetime import datetime
import math
from cycler import cycler
import numpy as np
import os
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from gym_pybullet_drones.examples.USV_trajectory import UsvTrajectory
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.plots_generation import PlotGeneration


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

os.environ["QT_QPA_PLATFORM"] = "wayland"
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.optimize import minimize
import autograd.numpy as np
from autograd import grad
from IPython.display import HTML, display
filename = 'results/OPT_500_20HZ'
df = np.load(filename + '/evaluations.npz')

plt.rc('font', size=25)
plt.rc('axes', titlesize=25)
plt.rc('axes', labelsize=25)
plt.rc('legend', fontsize=25)
plt.rc('figure', titlesize=1000)
plt.plot(df['timesteps']/1000, df['results'])
plt.title('Эффективность обучения алгоритма PPO')
plt.xlabel('Число эпизодов')
plt.ylabel('Сумарная награда за эипизод')
plt.show()


xyz1 = np.array([[0, 40, 0], [0, 60, 0], [0, 80, 0], [0, 100, 0]])
phi = np.array([np.random.uniform(-math.pi, math.pi),
                np.random.uniform(-math.pi, math.pi),
                np.random.uniform(-math.pi, math.pi),
                np.random.uniform(-math.pi, math.pi)])
time_data = TimeData(30, 300)

trajs = UsvTrajectory(time_data, m=4, r0=xyz1[:, 0:2], xyz0=xyz1, φ0=phi)
PLOT_FS = 20

trajs_s = trajs.sample(PLOT_FS)
usv_coord = trajs_s.xyz
fig = plt.figure(figsize=(40, 20))
ax = fig.add_subplot(111)
plots_usv = []
PlotGeneration.created_plot(plots_usv, ax, trajs.m, usv_coord, "БПНА", 3.0)
ax.set_xlabel('  x, м')
ax.set_ylabel('  y, м')
ax.set_title('Траектории')
tr_min = np.min(usv_coord, axis=(0, 1))
tr_max = np.max(usv_coord, axis=(0, 1))
ax.set(xlim=[tr_min[0], tr_max[0]],
       ylim=[tr_min[1], tr_max[1]])
ax.legend(fontsize=10)

def update(frame):
    start_frame = max(0, frame - 100)

    PlotGeneration.update_animation(start_frame, frame, usv_coord, plots_usv)

    full_plots = plots_usv
    return full_plots


ani1 = animation.FuncAnimation(fig, update, frames=trajs_s.time.n, blit=True, interval=100)
ani1.save('animation4.mp4', writer='ffmpeg')
plt.show()

