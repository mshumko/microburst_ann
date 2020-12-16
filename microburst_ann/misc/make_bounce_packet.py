import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def bounce_packet(t, n_peaks, bounce_period, peak_fwhm, packet_decay_time):
    """
    Create a bouncing packet microburst with n_peaks number
    of peaks with a width given by the peak_fwhm, and the entire
    packet decays with a an exponential timescale of 
    packet_decay_time.

    Parameters
    ----------

    Returns
    -------
    y: np.array
        The count amplitudes of the decaying packet with the same
        shape as the time array t.
    """
    y = np.zeros_like(t)

    t0_arr = [2*peak_fwhm+i*bounce_period for i in range(n_peaks)]

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
    """
    exp_arg = -(t-t0)**2 / (2*sigma**2)
    return A*np.exp(exp_arg)

if __name__ == '__main__':
    t = np.arange(0, 3, 20E-3)

    y, exp = bounce_packet(t, 5, 0.25, 0.1, 1)


    # y = gaus(t, 10, 2, 2)
    plt.plot(t, y)
    plt.plot(t, exp)
    plt.show()


