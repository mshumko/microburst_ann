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
    catalog, and randomly copy over the HILT counts and save
    to a hdf5 file.
Merge_Counts
    This class uses the output from the above two classes to
    generate a csv file of microbursts and non-microbursts,
    randomly shuffled, and with a label (non-microburst=0, 
    microburst=1).
Visualize_Counts
    Visualize the counts that were outputted Copy_Microburst_Counts
    and Copy_Non_Microburst_Counts.
"""

import pathlib
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

import microburst_ann.config as config
import microburst_ann.misc.load_hilt_data as load_hilt_data


class Copy_Microburst_Counts:
    """
    For each good microburst in the catalog_name catalog, copy the 
    20 ms HILT counts data into a microburst dataset with times and
    counts.

    Methods
    -------
    loop
        Loop over the catalog times, get the HILT data, and save
        to self.microburst_counts pd.DataFrame.
    save_counts
        Saves the resulting HILT counts to a hdf5 file.
    _load_catalog
        Internal method to load the csv microburst catalog, assign a
        time index, and parse the time column into pd.Timestamp objects.

    Attributes
    ----------
    catalog: pd.DataFrame
        The microburst catalog used to copy over the counts. Generated
        by __init__.
    microburst_counts: pd.DataFrame
        Contains the HILT counts, centered on the microburst time, with 
        the same time index as catalog.
    hilt_obj: object
        The HILT object that contains the time-resolved HILT data, used 
        from load_hilt_data.
    """

    def __init__(self, catalog_name, width_s=1, min_r2=0.9):
        """
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
                try:
                    self.hilt_obj = load_hilt_data.Load_SAMPEX_HILT(t)
                    self.hilt_obj.resolve_counts_state4()
                except RuntimeError as err:
                    if 'The SAMPEX HITL data is not in order.' in str(err):
                        continue
                    raise

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

class Copy_Nonmicroburst_Counts(Copy_Microburst_Counts):
    """
    Look for time windows outside of a given microburst 
    catalog, and randomly copy over the HILT counts and save
    to a hdf5 file. The same number of random times are chosen
    as there are microburst in each day in the microburst
    catalog: for example if there are 5 microbursts on one day,
    there also will be 5 non-microburst times picked from that
    day as well. There may be better approaches, but this one 
    seems simplest.
    
    In this case, the r^2 filter is not applied to the microburst
    catalog, to avoid accidently adding microbursts, as misshaped
    as they are, to the non-microburst counts dataset.

    Methods
    -------
    loop
        Loop over the HILT data, pick N random times from M random days,
        check that the random time does not encompass a microburst, and
        save to a self.nonmicroburst_counts pd.DataFrame.
    save_counts
        Saves the resulting HILT counts to a hdf5 file.
    _load_catalog (inhertied)
        Internal method to load the csv microburst catalog, assign a
        time index, and parse the time column into pd.Timestamp objects.

    Attributes
    ----------
    catalog: pd.DataFrame
        The microburst catalog used to check that the non-microburst 
        counts are not near a detected microburst. This catalog is loaded
        by the inhertied __init__.
    nonmicroburst_counts: pd.DataFrame
        Contains the random HILT counts.
    hilt_obj: object
        The HILT object that contains the time-resolved HILT data, used 
        from load_hilt_data.
    """

    def __init__(self, catalog_name, width_s=1):
        """
        Calls __init__ from the Copy_Microburst_Counts class and calls 
        the method to generate the same number of non-microburst times
        as microburst times.
        """
        super().__init__(catalog_name, width_s=width_s)
        return 

    def loop(self, near_thresh_s=1):
        """
        Loops over microburst detections. For each detection, pick a 
        corresponding random time from that day and save it if it was not
        near a microburst.

        Parameters
        ----------
        None

        Returns
        -------
        self.nonmicroburst_counts: pd.DataFrame
            A dataframe containing the HILT count 2d array with a time index. 
            The 2d array is of shape nMicrobursts x nWindowPoints.
        """
        # Create empty arrays.
        prev_date = pd.Timestamp.min.date()
        self.nonmicroburst_counts = pd.DataFrame(
            data=-1*np.ones((self.catalog.shape[0], self.width_dp*2), dtype=int),
            )
        self.nonmicroburst_times = pd.DataFrame(
            data={
                'dateTime':np.full(self.catalog.shape[0], pd.Timestamp.min.date(), 
                                                        dtype=object)
            })
        # Save this as a reference to speed up the calculation.
        numerical_catalog_dates = date2num(self.catalog.index)

        # Loop over every microburst detection and find random HILT times that do
        # not coincide with microbursts.
        for i, t in enumerate(self.catalog.index):
            # Load the HILT data if that day is not loaded yet.
            if t.date() != prev_date:
                print(f'Loading HILT data from {t.date()}')
                prev_date = t.date()
                try:
                    self.hilt_obj = load_hilt_data.Load_SAMPEX_HILT(t)
                    self.hilt_obj.resolve_counts_state4()
                except RuntimeError as err:
                    if 'The SAMPEX HITL data is not in order.' in str(err):
                        continue
                    raise
            
            time_threshold_not_met = True
            n_tries = 0
            while time_threshold_not_met:
                # Pick random times from self.hilt_obj.hilt_resolved DataFrame until one time is 
                random_row = self.hilt_obj.hilt_resolved.sample()
                numeric_row_time = date2num(random_row.index)
                # If the minimum time difference between the randomly picked time and the
                # microburst times is greater than near_thresh_s, we will keep this time.
                # Otherwise, draw anothe random time and try again until the condition is 
                # satisfied.
                if np.min(np.abs(numerical_catalog_dates-numeric_row_time)) > near_thresh_s/86400:
                    time_threshold_not_met = False
                
                # Check if we're headed for an infinite while loop and exit if uncessefully 
                # tried to pick a random time 10 tries.
                if n_tries >= 10:
                    break
                n_tries += 1

            # Find the numerical index corresponding to the random time.
            idx = np.where(self.hilt_obj.hilt_resolved.index == random_row.index[0])[0][0]
            # Copy the HILT counts
            hilt_counts = self.hilt_obj.hilt_resolved.iloc[
                idx-self.width_dp:idx+self.width_dp
                ].counts.to_numpy().astype(int)
            self.nonmicroburst_counts.iloc[i, :] = hilt_counts
            self.nonmicroburst_times.iloc[i, 0] = random_row.index[0]
        return

    def save_counts(self, save_name):
        """
        After you run the loop() method, this method saves the 
        self.nonmicroburst_counts to a csv or a hdf5 file, depending 
        on the save_name file extension.

        Parameters
        ----------
        save_name: str or pathlib.Path
            The name of the save file with the extension csv or h5. 
            No other save formats are currently supported.
        """
        self.nonmicroburst_counts.index = self.nonmicroburst_times['dateTime']
        # Set the index.
        save_path = pathlib.Path(config.PROJECT_DIR, 'data', save_name)
        if save_path.suffix == '.csv':
            self.nonmicroburst_counts.to_csv(save_path)
        elif save_path.suffix == '.h5':
            self.nonmicroburst_counts.to_hdf(save_path, 
                                        mode='w', key='counts')
        else:
            raise ValueError(
                f'save_path={save_name} must have a csv or h5 extension.'
                )
        return


class Visualize_Counts:
    """
    Given a csv file, visualize the microburst and nonmicroburst HILT
    counts. If there 

    Methods
    -------
    _load_data
        Load the csv file containing the HILT microburst or nonmicroburst
        counts in each row. 

    Attributes
    ----------
    hilt_data: pd.DataFrame
        The HILT microburst and/or nonmicroburst data.
    """
    def __init__(self, data_file_name):
        """

        """
        self.data_file_name = data_file_name
        self._load_data()
        return

    def plot(self, nrows=5, ncols=5, seed=123, label=None):
        """

        """
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols)

        # if label == 0:
        #     plot_str = 'non-microburst'
        # elif label == 1:
        #     plot_str = 'microbursts'
        # elif label is None:
        #     assert 'label' in self.hilt_data.keys(), (
        #         'The HILT data has no label column.')

        random_rows = self.hilt_data.sample(nrows*ncols, random_state=seed)
        plt_index = 0

        for ax_row in ax:
            for ax_i in ax_row:
                ax_i.plot(random_rows.iloc[plt_index, :], c='k')
                ax_i.text(
                    0, 1.1, 
                    (f'{random_rows.index[plt_index].date()}\n'
                    f'{random_rows.index[plt_index].time()}'), 
                    va='top', ha='left', transform=ax_i.transAxes, fontsize=5
                    )
                plt_index+=1
                ax_i.axis('off')

        plt.suptitle(f'ANN Training Data from\n{self.data_file_name}')
        plt.tight_layout()
        return ax

    def _load_data(self):
        """
        Loads the csv file containing a microburst or nonmicroburst
        HILT data in each row.
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None, add a hilt_data attribute pd.DataFrame to the object.
        """
        self.data_path = pathlib.Path(config.PROJECT_DIR, 'data', 
                                        self.data_file_name)
        self.hilt_data = pd.read_csv(self.data_path, index_col=0, 
                                    parse_dates=True)       
        return