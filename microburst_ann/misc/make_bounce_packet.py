import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def bounce_packet(t, n_peaks, peak_period, peak_fwhm, packet_decay_time):
    """
    Create a bouncing packet microburst with n_peaks, separated 
    by peak_period, of width peak_fwhm, and the whole packet decays
    with an exponential with the packet_decay_time time scale.

    The output counts are normalized to a max value of 1, intended to
    be scaled afterwards.

    Parameters
    ----------
    t: array-like
        A list or an array of integer of float timestamps
    n_peaks: int
        Number of Gaussian peaks
    peak_period: float
        The time interval between successive peaks. Usually is the
        electron bounce period, but can be anything.
    peak_fwhm: float
        The full width at half maximum (FWHM) of each peak. Assume
        that the peaks are all of the same width.
    packet_decay_time: float
        The exponential decay time scale for the packet. The counts
        decrease by 1/e after packet_decay_time. 

    Returns
    -------
    y: np.array
        The count amplitudes of the decaying packet with the same
        shape as the time array t. Normalized to a max value of 1.
    envelope: np.array
        The exponentially decaying count envelope with the same shape
        as the time array t. The envelope is normalized by the same 
        factor as y, therefore it is larger than 1 at t=0.
    """
    y = np.zeros_like(t)

    # Create an array of peak times, starting from 2*peak_fwhm,
    # with each successive peak a peak_period away from the first.
    t0_arr = [2*peak_fwhm+i*bounce_period for i in range(n_peaks)]

    # Iterate over the peak parameters and superpose each successive
    # Gaussian to y.
    for t0 in t0_arr:
        y += gaus(t, 1, t0, peak_fwhm/2.35482)

    # Scale the whole packet by the packet_decay_time.
    envelope = np.exp(-t/packet_decay_time)
    y *= envelope

    # Normalize the max amplitude to 1.
    envelope /= np.max(y)
    y /= np.max(y) 
    return y, envelope

def gaus(t, A, t0, sigma):
    """
    The gaussian function evaluated at times t.

    Parameters
    ---------
    t: array-like
        A list or an array of integer or float timestamp.
    A: float
        Amplitude of the Gaussian function
    t0: float
        Peak time of the Gaussian, in integer or float timestamp.
    sigma: float
        The standard deviation of the Gaussian.
    """
    exp_arg = -(t-t0)**2 / (2*sigma**2)
    return A*np.exp(exp_arg)

if __name__ == '__main__':
    t = np.arange(0, 3, 20E-3)

    y, exp = bounce_packet(t, 5, 0.25, 0.1, 1)

    plt.plot(t, y)
    plt.plot(t, exp)
    plt.show()