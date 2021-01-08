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
import pathlib
from datetime import date, datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal
import spacepy
import IRBEM

import wavelet_analysis
import make_bounce_packet

### Parameters ###
detrend=True
day = date(2015, 2, 2)
time_range = [datetime(2015, 2, 2, 6, 12, 0), 
            datetime(2015, 2, 2, 6, 13, 0)]
filter_time_range = True
sc_id = 4

### Load the FIREBIRD-II microburst counts ###
data_paths = sorted(list(pathlib.Path(
    f'/home/mike/research/firebird/Datafiles/'
    f'FU_{sc_id}/hires/level2'
    ).glob(f'FU{sc_id}_Hires_{day}*.txt')))
assert len(data_paths) == 1, 'FIREBIRD data not found.'
data_path = data_paths[0]
d = spacepy.datamodel.readJSONheadedASCII(str(data_path))
d['Time'] = pd.to_datetime(d['Time'])
cadence = float(d.attrs['CADENCE'])
counts = d['Col_counts'][:, 0]

if detrend:
    counts = pd.DataFrame(data=counts)
    background = counts.rolling(int(1/cadence)).quantile(0.50)
    counts -= background
    counts[np.isnan(counts)] = 0
    counts = counts.to_numpy().flatten()

if filter_time_range:
    idt = np.where((d['Time'] > time_range[0]) & (d['Time'] < time_range[1]))[0]
    counts = counts[idt]
    roll_n = 10
    counts = pd.DataFrame(data=counts).rolling(roll_n).mean()
    counts = np.roll(counts, -roll_n//2).flatten()
    counts[np.isnan(counts)] = 0
    time = d['Time'][idt]

    d['Alt'] = d['Alt'][idt]
    d['Lat'] = d['Lat'][idt]
    d['Lon'] = d['Lon'][idt]

    if 'background' in locals():
        background = background.iloc[idt]

# ### Calculate the wavelet power spectrum ###
w = wavelet_analysis.WaveletDetector(counts, time, 
    cadence, siglvl=0.95,
    mother='MORLET', dj=0.125, j1=60,
    wavelet_param=4*np.pi)
w.waveletTransform()

# Calculate the bounce period.
n_downsample=1
m = IRBEM.MagFields(kext='OPQ77')
downsampled_t = time[::n_downsample]
tb = -1*np.ones(downsampled_t.shape[0], dtype=float)
z = zip(downsampled_t, d['Alt'], d['Lat'], d['Lon'])

for i, (t, alt, lat, lon) in enumerate(z):
    try:
        tb[i] = m.bounce_period({'time':t, 'x1':alt, 'x2':lat, 'x3':lon}, None, 200)
    except ValueError as err:
        if (('Mirror point below the ground!' in str(err)) or 
            ('This is an open field line!' in str(err))):
            tb[i] = np.nan
        else:
            raise


# ### Make the plot ###
if detrend:
    n_panels = 3
else:
    n_panels = 2
fig, ax = plt.subplots(n_panels, 1, figsize=(6,8), sharex=True)
ax[0].plot(time, counts, 'k')

if detrend:
    ax[0].plot(time, background, 'r')

w.plotPower(ax=ax[-1])
ax[-1].plot(downsampled_t, tb, 'r')
# ax[1].axhline(peak_period, c='k', zorder=10)
plt.show()