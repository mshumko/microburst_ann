"""
This module uses a microburst catalog csv from my other study
sampex_microburst_widths to create a microburst training,
test, and validation data sets. The non-microburst times
are randomly picked from the SAMPEX data.

Classes
-------

"""

import pathlib

import pandas as pd
import h5py

from . import config

class Create_Microburst_Dataset:
    def __init__(self, catalog_name, width_s=3):
        """
        Create a dataset with microburst times and microburst counts.

        Parameters
        ----------

        Methods
        -------

        """
        return