# trajectory analysis
import numpy as np
import h5py
from scipy.fftpack import fft, ifft, ifftshift
from numba import njit

__all__ = [
    "readtrj",
    "correlate",
    "autocorrelation",
    "get_vacf",
    "mem_from_vacf",
]


def readtrj(trjfile, idx=[], x=False, v=True, f=False):
    """Reads a MD trajectory from h5md format (LAMMPS).
      In the context of this lib, only the velocities are needed.
      Might be extended, if required.

    Args:
        trjfile (str): path/to/traj.h5

    Returns:
        tuple : tuple of arrays, containing positions, velocities ,forces and time in this order. Not requested data is left out.
    """
    hf = h5py.File(trjfile, "r")
    time = hf["particles"]["all"]["velocity"]["time"][::]
    if v:
        if len(idx) == 0:
            vel = hf["particles"]["all"]["velocity"]["value"][:, :, :]
        else:
            vel = hf["particles"]["all"]["velocity"]["value"][:, idx, :]
        if not x and not f:
            return vel, time
        elif x and f:
            if len(idx) == 0:
                force = hf["particles"]["all"]["force"]["value"][:, :, :]
                pos = hf["particles"]["all"]["positions"]["value"][:, :, :]
                image = hf["particles"]["all"]["image"]["value"][:, :, :]
            else:
                force = hf["particles"]["all"]["force"]["value"][:, idx, :]
                pos = hf["particles"]["all"]["positions"]["value"][:, idx, :]
                image = hf["particles"]["all"]["image"]["value"][:, idx, :]
            return pos, image, vel, force, time
        elif x and not f:
            if len(idx) == 0:
                pos = hf["particles"]["all"]["positions"]["value"][:, :, :]
                image = hf["particles"]["all"]["image"]["value"][:, :, :]
            else:
                pos = hf["particles"]["all"]["positions"]["value"][:, idx, :]
                image = hf["particles"]["all"]["image"]["value"][:, idx, :]
            return pos, image, vel, time
        elif f and not x:
            if len(idx) == 0:
                force = hf["particles"]["all"]["force"]["value"][:, :, :]

            else:
                force = hf["particles"]["all"]["force"]["value"][:, idx, :]

            return vel, force, time

    elif x:
        if len(idx) == 0:
            pos = hf["particles"]["all"]["positions"]["value"][:, :, :]
            image = hf["particles"]["all"]["image"]["value"][:, :, :]
        else:
            pos = hf["particles"]["all"]["positions"]["value"][:, idx, :]
            image = hf["particles"]["all"]["image"]["value"][:, idx, :]
        if f:
            if len(idx) == 0:
                force = hf["particles"]["all"]["force"]["value"][:, :, :]

            else:
                force = hf["particles"]["all"]["force"]["value"][:, idx, :]

            return pos, image, force, time
        else:
            return pos, image, time
    elif f:
        if len(idx) == 0:
            force = hf["particles"]["all"]["force"]["value"][:, :, :]

        else:
            force = hf["particles"]["all"]["force"]["value"][:, idx, :]
        return force, time
    else:
        raise ValueError("At least one of x/v/f has to be set to True!")


def correlate(x, y):
    """Calculates the (cross-)correlation function of two time series through
    the Wiener–Khinchin theorem using fast Fourier transforms.

    Args:
        x (numpy.ndarray): 1d array of time series
        y (numpy.ndarray): 1d array of time series

    Returns:
        numpy.ndarray : time-correlation function between x and y
    """
    xp = ifftshift((x))
    yp = ifftshift((y))
    (n,) = xp.shape
    fx = fft(xp)
    fy = fft(yp)
    p = fx * np.conj(fy)
    pi = ifft(p)
    return np.real(pi)[: n // 2] / (np.arange(n // 2)[::-1] + n // 2)


def autocorrelation(x):
    """Calculates the (cross-)correlation function of two time series through
    the Wiener–Khinchin theorem using fast Fourier transforms.

    Args:
        x (numpy.ndarray): 1d array of time series

    Returns:
        numpy.ndarray : time-correlation function between x and x
    """
    xp = ifftshift((x))
    (n,) = xp.shape
    xp = np.r_[xp[: n // 2], xp[n // 2 :]]
    f = fft(xp)
    p = np.absolute(f) ** 2
    pi = ifft(p)
    return np.real(pi)[: n // 2] / (np.arange(n // 2)[::-1] + n // 2)


def get_vacf(trjfile, out_file, idx=[]):
    """Calculates the velocity auto-correlation function (VACF) from a trajectory in h5md format.


    Args:
        trjfile (str):  path/to/traj.h5
        out_file (str): path/to/vacf (Filename where to store vacf)
        idx (list or np.array): list of  atom indices to read from trajectory
    """

    # When default, assume that all beads are equivalent
    if len(idx) == 0:
        vel, time = readtrj(trjfile)
    # When specific list of indices is provided, work on subset of beads
    else:
        vel, time = readtrj(trjfile, idx)

    natoms = len(vel[0, :, 0])
    n = len(vel) // 2
    vacf = np.zeros(n)
    dt = time[1] - time[0]
    # Make sure, that the time[0] = 0
    time = np.arange(n) * dt
    for i in range(len(time)):
        time[i] = i * dt
    # sum and average over all atoms
    for i in range(natoms):
        vacf += autocorrelation(vel[:, i, 0])
        vacf += autocorrelation(vel[:, i, 1])
        vacf += autocorrelation(vel[:, i, 2])
    vacf = vacf / natoms / 3
    vacf = np.array([time[: len(vacf)], vacf])
    vacf = vacf.T

    np.savetxt(out_file, vacf, fmt="%6.6e")


def mem_from_vacf(vacf_in, n, step, G_file=""):
    """Evaluate integrated memory kernel from velocity auto correlation function.
    Wraps jitted _mem_from_vacf(...) for performance.

    Args:
        vacf_in (str, numpy.ndarray): File name or array of VACF to evaluate. shape=(number of time steps, 2)
        n (int): Final timestep to consider
        step (_type_): Takes into account every "step" value (considers vacf[:n,step])
        G_file (str, optional): Path/to/file to write results. Defaults to "imem".

    Returns:
        _type_: _description_
    """
    if type(vacf_in) == str:
        vacf = np.loadtxt(vacf_in)[:n:step]
        if G_file:
            out_f = G_file
        else:
            out_f = vacf_in + "_imem"
    else:
        vacf = vacf_in[:n:step]
        if G_file:
            out_f = G_file
        else:
            out_f = "imem"

    memory = np.zeros(vacf.shape)
    memory[:, 0] = vacf[:, 0]
    _mem_from_vacf(vacf, memory)
    np.savetxt(out_f, memory)
    return memory[:, 1]


@njit
def _mem_from_vacf(vacf, memory):
    """Evaluate the integrated single particle memory kernel from a velocity auto-correlation function.
    Implements the predictor-corrector variant as described in  Commun Phys 3, 126 (2020).


    Args:
        vacf (numpy.ndarray): File name or array of VACF to evaluate. shape=(number of time steps, 2). The time-step is evaluated from the 0th axis.
        memory (numpy.ndarray): Empty array to store results.
    """
    dt = vacf[1, 0] - vacf[0, 0]

    for i in range(1, len(vacf) - 1):
        memory[i, 1] = (1.0 - vacf[i, 1] / vacf[0, 1]) / (0.5 * dt)
        for j in range(1, i):
            memory[i, 1] += -2 * memory[j, 1] * vacf[i - j, 1] / vacf[0, 1]
        temp = (1.0 - vacf[i + 1, 1] / vacf[0, 1]) / (0.5 * dt)
        for j in range(1, i + 1):
            temp += -2 * memory[j, 1] * vacf[i + 1 - j, 1] / vacf[0, 1]
        memory[i, 1] = (memory[i - 1, 1] + 3.0 * memory[i, 1] + temp) / 5.0
    memory[len(memory) - 1, 1] = temp


################
