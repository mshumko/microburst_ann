"""
This scipt provides the interface to the classes in the prep_data.py
program. There are three boolean flags that control what script is run.

Parameters
----------
run_microburst_counts: bool
    Flag to run the microburst_counts class and generate 
    microburst_counts.csv file.
run_non_microburst_counts: bool
    Flag to run the nonmicroburst_counts class and generate
    nonmicroburst_counts.csv file.
"""
import pathlib
import time

import pandas as pd
import numpy as np

from microburst_ann.misc import prep_data

# Boolean flags
run_microburst_counts = False
run_non_microburst_counts = True


if run_microburst_counts:
    # Save the microburst counts to a csv file and optionally to
    # a hdf5 file (hdf5 data compression appears to be very crappy).
    cp = prep_data.Copy_Microburst_Counts('microburst_catalog_01.csv')
    cp.loop()
    # cp.save_counts('microburst_counts.h5')
    cp.save_counts('microburst_counts.csv')

if run_non_microburst_counts:
    cp = prep_data.Copy_Nonmicroburst_Counts('microburst_catalog_01.csv')
    try:
        cp.loop()
    finally:
        cp.save_counts('nonmicroburst_counts.csv')