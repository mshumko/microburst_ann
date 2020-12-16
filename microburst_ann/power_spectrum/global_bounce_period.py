"""
This script calculates the bounce period of locally-miroring 
electrons with an energy E.

Parameters
----------
E_kev: int
    Electron energy in units of keV.
alt_km: float
    Local mirror point altitude.
lat_bins/lon_bins: list-like
    The latitude and longitude bins used to calculate the bounce
    period.
"""

import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IRBEM

from microburst_ann import config

E_kev = 1000
alt_km = 500
time = '2000-01-01T00:00:00'

lat_bins = np.arange(-90, 90, 5)
lon_bins = np.arange(-180, 180, 5)

tb = pd.DataFrame(
    data=np.nan*np.ones((lat_bins.shape[0], lon_bins.shape[0])),
    index=lat_bins, columns=lon_bins
    )

m = IRBEM.MagFields(kext='OPQ77')

for i, lat in enumerate(lat_bins):
    for j, lon in enumerate(lon_bins):
        X = {'time':time, 'x1':alt_km, 'x2':lat, 'x3':lon}
        maginput = None
        try:
            tb.iloc[i,j] = m.bounce_period(X, maginput, E_kev)
        except ValueError as err:
            if (
                ('This is an open field line!' == str(err)) or 
                ('Mirror point below the ground!' in str(err))
                ):
                continue
            raise

tb.to_csv(config.PROJECT_DIR / 'data' / 'electron_tb_1mev.csv')

fig, ax = plt.subplots()
p = ax.pcolormesh(lon_bins, lat_bins, tb.to_numpy())
plt.colorbar(p)

# Overlay L-shell contours
L_lons = np.load('/home/mike/research/mission_tools/mission_tools'
                '/misc/irbem_l_lons.npy')
L_lats = np.load('/home/mike/research/mission_tools/mission_tools'
                '/misc/irbem_l_lats.npy')
L = np.load('/home/mike/research/mission_tools/mission_tools'
                '/misc/irbem_l_l.npy')
levels = [4,8]
CS = ax.contour(L_lons, L_lats, L, levels=levels, colors='w', linestyles='dotted')
plt.clabel(CS, inline=1, fontsize=10, fmt='%d')

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()