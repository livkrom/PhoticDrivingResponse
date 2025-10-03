"""
Module patients selects patientfiles, loads their EEG data and filters them.
"""
import argparse
from pathlib import Path
import numpy as np
import mne
from mne.io.base import BaseRaw

def parse_args()-> tuple[dict[str, str], argparse.Namespace]:
    """
    Parse command-line arguments for trial and time selection. 
    It assumes that patientdata is ordened in a time map, within a trial map.

    Returns
    -------
    :trial_map: dict of str
        Mapping from trial keys ("t0", "t1", "t2") to their corresponding folder names.
    :args: argparse.Namespace
        Parsed arguments with attributes:
        - trial : str
            Selected trial key.
        - time : str
            Selected time key (validated against the trial).
    """
    parser = argparse.ArgumentParser()

    trial_map = {
        "t0": "T0_T1_T2",
        "t1": "T0_T1",
        "t2": "T0_T1"}
    parser.add_argument("-tr", "--trial", choices = trial_map.keys(), default="t0", help = "Choose between t0, t1, t2")

    time_map = {
        "t0": ["t0", "t1", "t2"],
        "t1": ["t0", "t1"],
        "t2": ["t0", "t1"]
    }
    parser.add_argument("-t", "--time", choices = time_map.keys(), default="t1",
                        help = "Choose point in time: t0, t1 or t2")

    args = parser.parse_args()
    if args.time not in time_map[args.trial]:
        raise ValueError(f"Error: timepoint in {args.time} is not valid vor trial {args.trial}")

    return trial_map, args

def patient_files(trial_map: dict[str, str], args: argparse.Namespace) -> list[Path]:
    """
    Returns a list of patient files for a given trial/time combo.

    Parameters
    ----------
    :trial_map: dict
        Dictionary mapping trial arguments to folder names.
    :args: argparse
        Parsed command-line arguments, must contain `trial` and `time`.

    Returns
    -------
    :list: list
        List containing all EEG files in format .cnt within the folder. 
    """
    data_folder = Path("/Volumes/Docs/Bruikbare Data") / trial_map[args.trial] / args.time
    return list(data_folder.glob("*.cnt"))

def eeg(src, passband, notch = 50, plot: bool = False)-> BaseRaw:
    """
    Loads the EEG data.

    Parameters
    ----------
    :src: Path
        Pathway to EEG-data file.
    :passband: 1x2 list
        List containing the lower and upper frequency boundary of the passband filter.
    :notch: float
        Line frequency during the measurement to notch-filter for.
    :plot: bool, optional
        Option to plot the filtered raw EEG data with basic line- and passbandfiltering.

    Returns
    -------
    :raw: mne.BaseRaw
        Raw EEG data with basic line- and passbandfiltering.
    """
    raw = mne.io.read_raw_ant(src, preload=True, verbose='ERROR')
    raw.filter(l_freq=passband[0], h_freq=passband[1], picks="eeg", verbose='ERROR')

    line_freq = notch if (freq := raw.info["line_freq"]) is None else freq
    lowpass = np.arange(line_freq, raw.info["lowpass"]+1, line_freq)
    raw.notch_filter(freqs=(lowpass), notch_widths=(lowpass)/line_freq, picks=["eeg"], verbose='ERROR')

    if plot:
        raw.plot(scalings = "auto", title="Filtered EEG data", show=True, block=True)

    return raw
