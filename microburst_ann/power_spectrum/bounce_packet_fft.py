"""
This script calculates the Fourier phase space density (PSD)
of a synthetic bouncing packet counts with and without noise.

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
import scipy.fft
import scipy.signal

import make_bounce_packet

### Parameters ###
cadence=0.02
n_peaks=5
peak_period=0.5
peak_fwhm=0.1
packet_decay_time=1
t0_offset=1.5
scale_microburst_counts=500

y_intercept=5
slope=100

noise=True

# Calculate the peak frequency and +/- peak_freq_thresh around it.
peak_freq = 1/peak_period
peak_freq_thresh = 0.25
peak_freq_range = [peak_freq*(1-peak_freq_thresh), peak_freq*(1+peak_freq_thresh)]

### Generate the microburst counts ###
t = np.arange(0, 5, cadence)

y, _ = make_bounce_packet.bounce_packet(t, n_peaks, peak_period, peak_fwhm, 
                    packet_decay_time, t0_offset=t0_offset)
y *= scale_microburst_counts
y += y_intercept + t*slope

if noise:
    y = np.random.poisson(y)

### Calculate the FFT
detrended_y = scipy.signal.detrend(y, axis=0, type='linear')
psd = np.abs(scipy.fft.rfft(detrended_y))
freq = scipy.fft.rfftfreq(len(y), d=cadence)


### Plot the results ###
fig, ax = plt.subplots(3, 1, figsize=(6,8))
ax[0].plot(t, y)
ax[1].plot(t, detrended_y)
ax[2].axvspan(peak_freq_range[0], peak_freq_range[1],  alpha=0.5, color='red')
ax[2].axvline(1/peak_period, c='k')
ax[2].plot(freq, psd)


# Plot labels
ax[0].set(xlabel='Time [s]', ylabel='Original counts')
ax[1].set(xlabel='Time [s]', ylabel='Detrended counts')
ax[2].set(xlabel='Frequency [Hz]', ylabel='PSD', xlim=(0, 10))

plt.tight_layout()
plt.show()