"""
This module uses a microburst catalog csv from my other study
sampex_microburst_widths to create a microburst training,
test, and validation data sets. The non-microburst times
are randomly picked from the SAMPEX data.

Classes
-------
Copy_Microburst_Counts
    For each good microburst in a given microburst catalog,
    copy over the HILT counts within a time window and save
    to a hdf5 file.
Copy_Non_Microburst_Counts
    Look for time windows outside of a given microburst 
    catalog, to randomly copy over the HILT counts and save
    to a hdf5 file.
Visualize_Counts
    Visualize the counts that were outputted Copy_Microburst_Counts
    and Copy_Non_Microburst_Counts.
"""

import pathlib

import pandas as pd

import microburst_ann.config as config
import microburst_ann.misc.load_hilt_data as load_hilt_data


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
        self._load_catalog()
        self.width_dp = int(width_s/(2*20E-3))
        return

    def _load_catalog(self):
        """
        Loads the catalog using self.catalog_name.
        """
        self.catalog_path = pathlib.Path(config.PROJECT_DIR, 'data', 
                                        self.catalog_name)
        self.catalog = pd.read_csv(self.catalog_path, index_col=0, 
                                    parse_dates=True)                                
        return


if __name__ == "__main__":
    cp = Copy_Microburst_Counts('microburst_catalog_01.csv')