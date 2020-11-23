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
import numpy as np

import microburst_ann.config as config
import microburst_ann.misc.load_hilt_data as load_hilt_data


class Copy_Microburst_Counts:
    def __init__(self, catalog_name, width_s=1, min_r2=0.9):
        """
        For each good microburst in the catalog_name catalog, copy the 
        20 ms HILT counts data into a microburst dataset with times and
        counts.

        Parameters
        ----------
        catalog_name: str
            The name of the microburst catalog to load.
        width_s: float, optional
            The time width, centered on the microburst, to copy over the 
            counts into the microburst_counts dataset. This class assumes
            20 ms HILT candence.
        min_r2: float, optional
            The minimum R^2 value to filter the data by. This makes sure
            that the microbursts are well-shaped.

        Methods
        -------
        loop
            Loop over the catalog times, get the HILT data, and save
            to self.microburst_counts pd.DataFrame.
        save_counts
            Saves the resulting HILT counts to a hdf5 file.
        """
        self.catalog_name = catalog_name
        self.min_r2 = min_r2
        self.width_dp = int(width_s/(2*20E-3))

        self._load_catalog()
        return

    def loop(self):
        """
        Loops over every good microburst detection and copy over the HILT
        counts.

        Parameters
        ----------
        None

        Returns
        -------
        self.microburst_counts: pd.DataFrame
            A dataframe containing the HILT count 2d array with the microburst
            time index. The 2d array is of shape nMicrobursts x nWindowPoints.
        """
        prev_date = pd.Timestamp.min.date()
        self.microburst_counts = pd.DataFrame(
            data=-1*np.ones((self.catalog.shape[0], self.width_dp*2), dtype=int),
            index=self.catalog.index
            )

        for t, row in self.catalog.iterrows():
            # Load the HILT data if that day is not loaded yet.
            if t.date() != prev_date:
                print(f'Loading HILT data from {t.date()}')
                prev_date = t.date()
                self.hilt_obj = load_hilt_data.Load_SAMPEX_HILT(t)
                self.hilt_obj.resolve_counts_state4()

            # Find the numerical index corresponding to t.
            idx = np.where(self.hilt_obj.hilt_resolved.index == t)[0][0]
            # Copy the HILT counts
            hilt_counts = self.hilt_obj.hilt_resolved.iloc[
                idx-self.width_dp:idx+self.width_dp
                ].counts.to_numpy().astype(int)
            self.microburst_counts.loc[t, :] = hilt_counts
        return self.microburst_counts

    def save_counts(self, save_name):
        """
        After you run the loop() method, this method saves the 
        self.microburst_counts to a csv or a hdf5 file, depending 
        on the save_name file extension.

        Parameters
        ----------
        save_name: str or pathlib.Path
            The name of the save file with the extension csv or h5. 
            No other save formats are currently supported.
        """
        save_path = pathlib.Path(config.PROJECT_DIR, 'data', save_name)
        if save_path.suffix == '.csv':
            self.microburst_counts.to_csv(save_path)
        elif save_path.suffix == '.h5':
            self.microburst_counts.to_hdf(save_path, 
                                        mode='w', key='counts')
        else:
            raise ValueError(
                f'save_path={save_name} must have a csv or h5 extension.'
                )
        return

    def _load_catalog(self):
        """
        Loads the catalog using self.catalog_name and adds a 
        self.catalog object attribute, a pd.DataFrame
        object that contains the microburst detections.

        Parameters
        ----------
        None

        Returns
        -------
        None, 
        """
        self.catalog_path = pathlib.Path(config.PROJECT_DIR, 'data', 
                                        self.catalog_name)
        self.catalog = pd.read_csv(self.catalog_path, index_col=0, 
                                    parse_dates=True)       
        self.catalog = self.catalog[self.catalog.r2 >= self.min_r2]
        return


if __name__ == "__main__":
    cp = Copy_Microburst_Counts('microburst_catalog_01.csv')
    cp.loop()
    cp.save_counts('microburst_counts.h5')
    cp.save_counts('microburst_counts.csv')
