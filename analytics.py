"""
Analytics
"""
from pathlib import Path
from scipy.stats import friedmanchisquare
import re
import matplotlib.pyplot as plt
import pandas as pd


def stats_base(power_path: Path, time_map, args, paired: bool = True):
    """
    Checks if there is a difference in baseline frequencies.
    In this code we average over all given baseline channels. 

    Parameters
    ----------
    :files: List[Path]
        List containing all Path objects for successfully processed files.
    :trial_map: dict
        Dictionary mapping trial arguments to folder names.
    :args: argparse
        Parsed command-line arguments, must contain `trial` and `time`.
    :paired: bool
        For now we are testing on paired data, but this will not always be the case. 
    """
    files = list(power_path.glob("*_power.pkl"))
    baselines = {}

    # Averaging over baselines
    for file in files:
        match = re.match(r"VEP(\d+)_(\d+)_power.pkl", file.name)
        if not match:
            print(f"Unexpected filename in folder: {file.name}")
            continue
        patient, timepoint = match.groups()
        patient, timepoint = int(patient), int(timepoint)

        df = pd.read_pickle(file)
        channels = [c for c in df.columns if c.endswith("_BASE")]
        df['Baseline_avg'] = df[channels].mean(axis=1)
        df = df[["Frequency", "Harmonic", "Baseline_avg"]].copy()

        if patient not in baselines:
            baselines[patient] = {}
        baselines[patient][timepoint] = df

    if args.time == "all":
        args.time = time_map[args.trial]
    timepoints = args.time if isinstance(args.time, list) else [args.time]
    
    # Defining stimulation & harmonic pairs
    freqs = set()
    for p_data in baselines.values():
        for tp, df in p_data.items():
            pairs = list(zip(df["Frequency"], ["Harmonic"]))
            freqs.update(pairs)
    freqs = sorted(freqs)

    # Gathering data per frequency pair
    results = []
    for freq, harm in freqs:
        values_per_tp = []
        for tp in timepoints:
            values = []
            for patient, tp_dict in baselines.items():
                if tp in tp_dict:
                    df = tp_dict[tp]
                    sel = df[(df["Frequency"] == freq) & (df["Harmonic"] == harm)]
                    if not sel.empty:
                        values.append(sel["Baseline_avg"].values[0])
            values_per_tp.append(np.array(values))
        
        k = len(timepoints)

    if paired:
        if k == 2:
            # wilcoxon signed rank test
        elif k > 2:
            #frieman chi squared test
        else: 
            raise ValueError ("Not enough baselines to compare.")
    else:
        #mann whitney u test 2 maps
    
def stats_power():
    