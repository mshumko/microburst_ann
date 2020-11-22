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
# import h5py

from microburs_ann import config
from microburs_ann.misc import load_hilt_data

class Copy_Microburst_Counts:
    def __init__(self, catalog_name, width_s=1):
        """
        For each good microburst in the catalog_name catalog, copy the 
        20 ms HILT counts data into a microburst dataset with times and
        counts.

        Parameters
        ----------
        catalog_name: str
            The name of the microburst catalog to load.
        width_s: float
            The time width, centered on the microburst, to copy over the 
            counts into the microburst_counts dataset. This class assumes
            20 ms HILT candence.

        Methods
        -------

        """
        self.catalog_name = catalog_name
        self.catalog_path = pathlib.Path(config.PROJECT_DIR, 'data', 
                                        self.catalog_name)
        self.width_dp = int(width_s/(2*20E-3))
        return

if __name__ == "__main__":
    cp = Copy_Microburst_Counts('microburst_catalog_01.csv')