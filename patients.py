"""
Module patients selects patientfiles, loads their EEG data and filters them.
"""
import re
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import mne
from mne.io.base import BaseRaw
mne.set_log_level('error')

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
        "t2": "T0_T2"}
    parser.add_argument("-tr", "--trial", choices = trial_map.keys(), default="t2", help = "Choose between t0, t1, t2")

    time_map = {
        "t0": ["t0", "t1", "t2"],
        "t1": ["t0", "t1"],
        "t2": ["t0", "t2"]
    }
    parser.add_argument("-t", "--time", choices = list(time_map.keys()), default="all",
                        help = "Choose point in time: t0, t1, t2 or all")

    args = parser.parse_args()
    if args.time == "all":
        args.time = time_map[args.trial]
    elif args.time not in time_map[args.trial]:
        raise ValueError(f"Error: timepoint in {args.time} is not valid vor trial {args.trial}")

    return trial_map, time_map, args

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
    data_folder = Path("/Volumes/Docs/Bruikbare Data") / trial_map[args.trial]
    timepoints = args.time if isinstance(args.time, list) else [args.time]

    files = []
    for tp in timepoints:
        folder = data_folder / tp
        files.extend(folder.glob("*.cnt"))  # collect all .cnt files

    return files

def eeg(src, passband, notch = 50, occi: bool = False, plot: bool = False)-> BaseRaw:
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
    :occi: bool, optional
            Option to choose either all channels or only the occipital ones.
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

    threshold = 1e-6
    channels_dropped = set(['EOG']+[ch for ch in raw.ch_names if np.all(np.abs(raw.get_data(picks=ch))<threshold)])
    if channels_dropped:
        raw.drop_channels([ch for ch in channels_dropped if ch in raw.ch_names])
        print(f"Dropped channels: {channels_dropped}")
    if occi:
        raw = raw.copy().pick(["O1", "O2", "Oz"])

    if plot:
        raw.plot(scalings = "auto", title="Filtered EEG data", show=True, block=True)

    return raw

def save_pickle_results(data, pt_file, folder_name, feat: str = "power"):
    """
    Saves dataframes (singles or multiple) as pickle files in designated pathway.

    Parameters
    ----------
    :data: pd.DataFrame | tuple | list
        Name of the patient file: VEPxx_T, xx = patient ID, T = timepoint.
    :pt_file: str
        Dataframe containing powers at different frequencies. 
    :folder_name:
        Name for the folder to save the pickle files. 
    :feat: str
        Type of feature that is being saved. Either "power" or "plv". 
    :verbose: bool, optional
        Option to print statements. 

    """
    path = Path(f"./{folder_name}")
    path.mkdir(parents=True, exist_ok=True)

    if isinstance(data, dict) and feat == "plv":
        for ending, item in data.items():
            single_path = path / f"{pt_file.stem}_{feat}_{ending}.pkl"
            item.to_pickle(single_path)
    else:
        single_path = path / f"{pt_file.stem}_{feat}.pkl"
        data.to_pickle(single_path)

    return

def filter_files(folder: str, time_map: dict, args, feat: str = "power"):
    """
    Filters patients with complete data across required timepoints for either power or PLV results.
    Moves incomplete patient files to ./incomplete_files.

    Parameters
    ----------
    :folder_power: str
        Name of the folder containing the power results.
    :trial_map: dict
        Dictionary mapping trial arguments to folder names.
    :args: argparse
        Parsed command-line arguments, must contain `trial` and `time`.
    :feat: str
        Feature type: "power" or "plv

    Returns 
    -------
        List of patient IDs with complete data across required timepoints.  
    """
    print(f"Finding complete patient datasets in {folder}...")

    path = Path(f"./{folder}")
    if feat == "power":
        files = list(path.glob("*_power.pkl"))
        file_pattern = r"(VEP\d+)_([123])_power"
        required_suffixes = ["_power.pkl"]
    elif feat == "plv":
        files = list(path.glob("*_plv_stim.pkl"))
        file_pattern = r"(VEP\d+)_([123])_plv_stim"
        required_suffixes = ["_plv_stim.pkl", "_plv_base.pkl"]
    else:
        raise ValueError("Feature type must be 'power' or 'plv'")

    if not files:
        print(f"No power files found in {path.resolve()}")
        return []

    if args.time == "all":
        args.time = time_map[args.trial]
    timepoints = args.time if isinstance(args.time, list) else [args.time]

    timepoint_mapping = {"1": "t0", "2": "t1", "3": "t2"}
    patient_times = defaultdict(set)

    for f in files:
        file = f.stem
        match = re.match(file_pattern, file)
        if not match:
            print(f"Unexpected filename: {file}")
            continue
        patient_id, time_num = match.groups()
        timepoint = timepoint_mapping.get(time_num)
        if timepoint:
            if feat == "power":
                patient_times[patient_id].add(timepoint)
            if feat == "plv":
                stim_file = path / f"{patient_id}_{time_num}_plv_stim.pkl"
                base_file = path / f"{patient_id}_{time_num}_plv_base.pkl"
                if stim_file.exists() and base_file.exists():
                    patient_times[patient_id].add(timepoint)

    complete = {pid for pid, tps in patient_times.items() if set(timepoints).issubset(tps)}

    all_files = []
    for suffix in required_suffixes:
        all_files.extend(path.glob(f"*{suffix}"))

    to_remove = [f for f in all_files if f.stem.split("_", 1)[0] not in complete]

    trash_folder = Path("./incomplete_files")
    trash_folder.mkdir(parents=True, exist_ok=True)
    for f in to_remove:
        f.rename(trash_folder / f.name)

    return sorted(complete)
