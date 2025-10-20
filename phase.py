"""
Module phase evaluates the phase-locking of the stimulation frequencies. 
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
class Phase:
    """ Implements the full phasey calculation pipeline for EEG data. """
    passband: list
    eeg: BaseRaw

    def run(self):
        """
        Run the full pipeline on EEg data and return the SNR DataFrame.
        
        :EEG: filtered EEG data, should contain trigger timing in annotations.
        """
        df_stim = self._stimulation_phase(self.eeg, base=False)
        filtered_epochs = self._epoch_phase(df_stim, self.eeg, upper_lim = 40)
        phases = self._fft_phase(filtered_epochs, occi=True, plot=False, save=False)

        return phases

    @staticmethod
    def _stimulation_phase(raw:BaseRaw, save: bool = False, base: bool = False) -> pd.DataFrame:
        """
        Extracts the stimulation blocks from trigger-annotations of EEG Data.

        Parameters
        ----------        
        :raw: mne.BaseRaw
            EEG data, should contain trigger timing in annotations.
        :save: bool, optional
            Option to save the made dataframe.
        :base: bool, optional
            Option to create baseline blocks.

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
        
        # Option to create baseline blocks
        if base:
            baseline_blocks = []
            df_epochs = df.loc[df.groupby("block")["sample"].idxmin()].reset_index(drop=True)
            df_epochs["ends"] = df.groupby("block")["sample"].max().values
            tmax = int(math.ceil(((df_epochs["ends"] - df_epochs["sample"]) / sfreq).min()))
            
            for rep, rep_epochs in df.groupby("rep"):
                for f in rep_epochs["freq"].unique():
                    if np.isnan(f):
                        continue
                    start = rep_epochs["sample"].min() - 0.1*sfreq - tmax*sfreq
                    period_samples = sfreq / f
                    baseline_samples = np.arange(start, start + (tmax * sfreq), period_samples)
                    used_samples = set()

                    for sample in baseline_samples:

                        # unieke offsets nodig MNE
                        event_sample = int(sample + rep)
                        while event_sample in used_samples:
                            event_sample += 1

                        used_samples.add(event_sample)
                        baseline_blocks.append({
                            "sample": int(event_sample), 
                            "previous": int(0),
                            "freq": int(f),
                            "rep": int(rep),
                        })

        # Option to save the dataframe
        if save:
            df.to_csv("phase_stimulation_info.csv", index=False)
            
            if base:
                df_base = pd.DataFrame(baseline_blocks)
                df_base.to_csv("baseline_info.csv", index=False)
                print("Stimulation dataframes saved as phase_stimulation_info.csv and baseline_info.csv")
            else:
                print("Stimulation dataframe saved as phase_stimulation_info.csv")

        if base:
            return df, df_base
        else:
            return df

    @staticmethod
    def _epoch_phase(df: pd.DataFrame, raw: BaseRaw, upper_lim: int = int(40)
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
        :baseline_blocks: ..., optional
            ...

        Returns
        -------
        :filtered_epochs: dict containing mne.Epochs
            Dictionary with keys (stim_freq, harmonic_freq) and values mne.Epochs objects.
        freqs : pd.DataFrame
            MultiIndex DataFrame listing all frequency-harmonic pairs.
        """
        threshold = 1e-6
        channels_dropped = set(['EOG'] + [ch for ch in raw.ch_names if np.all(np.abs(raw.get_data(picks=ch)) < threshold)])
        raw.drop_channels([ch for ch in channels_dropped if ch in raw.ch_names])
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
                raw_filt = raw_copy.filter(l_freq=l_freq, h_freq=h_freq, picks="eeg", phase='zero', verbose="ERROR")

                tmax = (cycles/h)+tmin
                epochs = mne.Epochs(raw_filt, np_epochs, event_id={str(f):f}, tmin=tmin, tmax=tmax,
                                baseline=None, preload=True, verbose="ERROR")

                filtered_epochs[(f,h)] = epochs
        filtered_epochs = pd.Series(filtered_epochs)
        filtered_epochs.index = pd.MultiIndex.from_tuples(filtered_epochs.index, names=["Frequency", "Harmonic"])            

        return filtered_epochs

    @staticmethod
    def _fft_phase(filtered_epochs: pd.MultiIndex, occi: bool = False, plot: bool = False,
                   save: bool = False)-> dict[tuple[float, float], dict[str, float]]:
        """
        Automatically compute FFT for all stimuli and get the phase.

        Parameters
        ----------
        :filtered_epochs: Pandas MultiIndex
            Multi-index with keys [stim_freq] and [harmonic_freq] and values mne.Epochs objects.
        :occi: bool, optional
            Option to choose either all channels or only the occipital ones.
        :plot: bool, optional
            Option to plot the phase-locking values. 
        :save: bool, optional
            Option to save the made dataframe.

        Returns
        -------
        :phases: dict
            Dictionairy containing the phase-locking values per (Freq, Harm)-tuple as [ch, phase]
       """
        angles = {}
        phases = {}

        for (f,h) in filtered_epochs.index:
            # Option to choose occipital channels only.
            if occi:
                ep = filtered_epochs[(f,h)].copy().pick(["O1", "O2", "Oz"])
            else:
                ep = filtered_epochs[(f,h)].copy().pick("eeg")

            ch_names = ep.info["ch_names"]
            sfreq = ep.info["sfreq"]
            data = ep.get_data()
            _, _, n_times = ep.get_data().shape

            window = np.hanning(n_times)
            data_windowed = data * window[np.newaxis, np.newaxis, :]
            fft_data = rfft(data_windowed, axis=2)

            freqs = rfftfreq(n_times, 1/sfreq)
            center_idx = np.argmin(np.abs(freqs - h))
            center_idx = np.clip(center_idx, 0, n_times - 1)

            angles[(f,h)] = np.angle(fft_data[:, :, center_idx])
            plv = np.abs(np.mean(np.exp(1j * angles[(f,h)]), axis=0))

            mean_plv = np.mean(plv)
            mean_phase = np.angle(np.mean(np.exp(1j * np.angle(np.mean(np.exp(1j * angles[(f, h)]), axis=0)))))

            phases[(f,h)] = {"angles": angles[(f,h)], 
                             "plv": dict(zip(ch_names, plv)), 
                             "ch_names": ch_names,
                             "mean_plv": mean_plv,
                             "mean_phase": mean_phase}

        if plot:
            colors = plt.get_cmap('tab10').colors
            n_pairs = len(angles)
            n_cols = math.ceil(math.log(n_pairs,2))
            n_rows = math.ceil(n_pairs/n_cols)
            fig, axes = plt.subplots(n_rows, n_cols, subplot_kw=dict(polar=True),
                             figsize=(5*n_cols, 10*n_rows))
            axes = axes.flatten()

            for idx, ((f,h), content) in enumerate(phases.items()):
                p = content["angles"]
                ax = axes[idx]
                ep = filtered_epochs[(f,h)]
                n_epochs, n_channels = p.shape

                ax.set_title(r'$p(\theta \mid f=%d\,Hz, h=%d)$' % (f,h), fontsize=10,
                             y = 0.0, x = -0.5, rotation = 90, ha = 'left', va='bottom')

                for ch_idx in range(n_channels):
                    ch_name = ch_names[ch_idx]
                    color = colors[ch_idx % len(colors)]
                    jitter = 1 + np.random.normal(-0.05, 0.05, size=n_epochs)

                    ax.scatter(p[:, ch_idx], jitter, s=5,
                            c=[color], alpha=0.7, label=ch_name if idx ==0 else None)

                mean_vector = np.mean(np.exp(1j * p.flatten()))
                mean_angle = np.angle(mean_vector)
                plv_mean = np.abs(mean_vector)
                ax.plot([0, mean_angle], [0, plv_mean], color='red', linewidth=3,
                        label='Mean vector' if idx == 0 else None)
                ax.set_ylim(0, 1.2)
                ax.set_rgrids([0.25, 0.5, 0.75, 1.0], angle=22.5, fontsize=5)

            for ax in axes [n_pairs:]:
                ax.axis('off')

            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.5))
            fig.suptitle('Phase distributions across frequencies and harmonics', fontsize=16)
            plt.subplots_adjust(top=0.85, bottom=0.10, left=0.08, right=0.92, hspace=0.5, wspace=0.2)
            plt.show()

        if save:
            rows = []
            for (f,h), ch_dict in phases.items():
                row = {'Frequency': f, 'Harmonic': h}

                for ch_idx, ch_name in enumerate(ch_dict["ch_names"]):
                    row[f"{ch_name}_plv"] = ch_dict["plv"][ch_name]
                    row[f"{ch_name}_angles"] = np.degrees(np.angle(np.mean(np.exp(1j * ch_dict["angles"][:, ch_idx]))))

                row["mean_plv"] = ch_dict["mean_plv"]
                row["mean_phase"] = np.degrees(ch_dict["mean_phase"])
                rows.append(row)

            df = pd.DataFrame(rows)
            df = df.sort_values(['Frequency','Harmonic'])
            df.to_csv('phases.csv', index=False)
            print("Saved phases-dataframe to phases.csv")

        return phases
