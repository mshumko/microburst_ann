"""
A command-line interface to plot the daily SAMPEX/HILT data 
"""
import time
from datetime import datetime
import argparse

import matplotlib.pyplot as plt

from sampex_microburst_widths.misc import load_hilt_data


parser = argparse.ArgumentParser(description=('This script plots the '
    'SAMPEX HILT data.'))
parser.add_argument('date', nargs=3, type=int,
    help=('This is the date to plot formatted as YYYY MM DD')) 
args = parser.parse_args()
date = datetime(*args.date)

start_time = time.time()
l = load_hilt_data.Load_SAMPEX_HILT(date)
l.resolve_counts_state4()
# a = load_hilt_data.Load_SAMPEX_Attitude(date)
print(f'Load time = {time.time()-start_time} s')

plt.plot(l.hilt_resolved.index[::10], l.hilt_resolved.counts[::10])
# plt.yscale('log')
plt.show()