"""
This script calculates the wavelet power spectrum of a synthetic 
bouncing packet counts with and without noise.

Parameters
----------
n_peaks: int
    Number of Gaussian peaks
scale_microburst_counts: float
    How much to scale the peak heights by.
peak_period: float
    The time interval between successive peaks. Usually is the
    electron bounce period, but can be anything.
peak_fwhm: float
    The full width at half maximum (FWHM) of each peak. Assume
    that the peaks are all of the same width.
packet_decay_time: float
    The exponential decay time scale for the packet. The counts
    decrease by 1/e after packet_decay_time. 
t0_offset: float
    The first peak time offset in time. 
y_intercept: float
    Adds a y-intercept to the count background.
slope: float
    Adds a slope to the count background.
noise: bool
    Toggle if noise should be added.
"""
import numpy as np
import matplotlib.pyplot as plt
# import scipy.fft
import scipy.signal

import wavelet_analysis
import make_bounce_packet

### Parameters ###
cadence=0.02
n_peaks=2
peak_period=0.5
peak_fwhm=0.08
packet_decay_time=1
t0_offset=5
scale_microburst_counts=800

y_intercept=100
slope=100

noise=True
detrend=True

# Calculate the peak frequency and +/- peak_freq_thresh around it.
peak_freq = 1/peak_period
peak_freq_thresh = 0.25
peak_freq_range = [peak_freq*(1-peak_freq_thresh), peak_freq*(1+peak_freq_thresh)]

### Generate the microburst counts ###
t = np.arange(0, 10, cadence)

y, _ = make_bounce_packet.bounce_packet(t, n_peaks, peak_period, peak_fwhm, 
                    packet_decay_time, t0_offset=t0_offset)
y *= scale_microburst_counts
y += y_intercept + t*slope

if noise:
    y = np.random.poisson(y)

if detrend:
    y = scipy.signal.detrend(y, type='linear')

### Calculate the wavelet power spectrum ###
w = wavelet_analysis.WaveletDetector(y, t, cadence, 
    mother='MORLET', dj=0.125, j1=50,
    wavelet_param=4*np.pi)
w.waveletTransform()

### Make the plot ###
fig, ax = plt.subplots(2, 1, figsize=(6,8), sharex=True)
ax[0].plot(t, y)
w.plotPower(ax=ax[1])
ax[1].axhline(peak_period, c='k', zorder=10)
plt.show()