"""
Module power calculates the coherency of the stimulation frequencies. 
"""
import math
from dataclasses import dataclass
import numpy as np
import pandas as pd

import mne
from mne.io.base import BaseRaw
import matplotlib
import matplotlib.pyplot as plt
from numpy.fft import rfft, rfftfreq
matplotlib.use("TkAgg")

@dataclass
class Coherency:
    """ Implements the full coherency calculation pipeline for EEG data. """
    passband: list
    eeg: BaseRaw

    def run(self):
        """
        Run the full pipeline on EEg data and return the SNR DataFrame.
        
        :EEG: filtered EEG data, should contain trigger timing in annotations.
        """
        df_stim = self._stimulation_coherency(self.eeg)
        filtered_epochs, freqs = self._epoch_coherency(df_stim, self.eeg, plot=False)

        return ...

    @staticmethod
    def _stimulation_coherency(raw:BaseRaw, save: bool = False) -> pd.DataFrame:
        """
        Extracts the stimulation blocks from trigger-annotations of EEG Data.

        Parameters
        ----------        
        :raw: mne.BaseRaw
            EEG data, should contain trigger timing in annotations.
        :save: bool, optional
            Option to save the made dataframe.

        Returns
        -------
        :df: Pandas DataFrame
            Dataframe containing EEG data information.
        """
        sfreq = raw.info["sfreq"]
        block_threshold = 1.0 * sfreq

        sample, _ = mne.events_from_annotations(raw)
        df = pd.DataFrame(sample, columns=["sample", "previous", "event_id"])
        df["block"] = (np.diff(np.r_[0, df["sample"].to_numpy()]) > block_threshold).cumsum()

        def compute_freq(samples):
            return int(round(1/np.mean(np.diff(samples)/sfreq),2)) if len(samples) > 1 else np.nan

        df["freq"] = df.groupby("block")["sample"].transform(compute_freq)

        rep, rep_freqs = 1, set()
        for block_id, freq in df.groupby("block")["freq"].first().items():
            if pd.isna(freq):
                continue
            if freq in rep_freqs:
                rep += 1
                rep_freqs = {freq}
            else:
                rep_freqs.add(freq)
            df.loc[df["block"] == block_id, "rep"] = int(rep)
        df.drop(["event_id"], axis=1, inplace=True)

        # Option to save the dataframe
        if save:
            df.to_csv("coherency_stimulation_info.csv", index=False)
            print("Stimulation dataframe saved as coherency_stimulation_info.csv")

        return df

    @staticmethod
    def _epoch_coherency(df: pd.DataFrame, raw: BaseRaw, upper_lim: int = int(40),
                         )-> tuple[dict[tuple[int, int], mne.Epochs], pd.DataFrame]:
        """
        Find epochs of the raw EEG data for different stimulation frequencies. 

        Parameters
        ----------
        :df: Pandas DataFrame
            Dataframe containing information about the stimulation blocks.
        :raw: mne.BaseRaw
            EEG data, should contain trigger timing in annotations.
        :upper_lim: integer
            Maximum frequency that we are interested in, default is in the lower end of the gamma-band.
        :save: bool, optional
            Option to save the made dataframe.

        Returns
        -------
        :filtered_epochs: dict containing mne.Epochs
            Dictionary with keys (stim_freq, harmonic_freq) and values mne.Epochs objects.
        freqs : pd.DataFrame
            MultiIndex DataFrame listing all frequency-harmonic pairs.
        """
        threshold = 1e-6
        channels_dropped = ['EOG'] + [ch for ch in raw.ch_names if np.all(np.abs(raw.get_data(picks=ch)) < threshold)]
        raw.drop_channels(channels_dropped)
        print(f"Dropped channels: {channels_dropped}")

        freqs_stim = df["freq"].unique()
        filtered_epochs = {}
        tmin = 0.05
        cycles = 1.5

        np_epochs = df[["sample", "previous", "freq"]].to_numpy(dtype=int)
        for f in freqs_stim:
            max_harm = math.floor(upper_lim/f)
            for i in range(1, max_harm + 1):
                h = f * i
                bandwidth = 1
                l_freq = h - bandwidth
                h_freq = h + bandwidth

                raw_copy = raw.copy()
                raw_filt = raw_copy.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", verbose="ERROR")

                tmax = (cycles/h)+tmin
                epochs = mne.Epochs(raw_filt, np_epochs, event_id={str(f):f}, tmin=tmin, tmax=tmax,
                                baseline=None, preload=True, verbose="ERROR")

                filtered_epochs[(f,h)] = epochs
        filtered_epochs = pd.Series(filtered_epochs)
        filtered_epochs.index = pd.MultiIndex.from_tuples(filtered_epochs.index, names=["Frequency", "Harmonic"])
        return filtered_epochs
    
    @staticmethod
    def _fft_coherency(filtered_epochs: pd.MultiIndex, occi: bool = False)->
        """
        Automatically compute FFT for all blocks.

        Parameters
        ----------
        :filtered_epochs: Pandas MultiIndex
            Multi-index with keys [stim_freq] and [harmonic_freq] and values mne.Epochs objects.
        :occi: bool, optional
            Option to choose either all channels or only the occipital ones. 
       """
        coherencies = {}
        
        for (f,h) in filtered_epochs.index:
            # Option to choose occipital channels only.
            if occi:
                ep = filtered_epochs.copy().pick(["O1", "O2", "Oz"])
            else:
                ep = filtered_epochs.copy().pick("eeg")

            # Convert stimulus signal

            
            # Hann window
            window = np.hanning(ep.shape[1])
            epoch_w = ep * window

            # FFT
            fft_data = np.fft.rfft(epoch_w, axis=1)

