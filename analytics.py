"""
Analytics
"""
import re
from pathlib import Path
from scipy.stats import wilcoxon, friedmanchisquare, mannwhitneyu, kruskal, chi2
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def stats_base(power_path: Path, paired: bool = True, save: bool = True):
    """
    Checks if there is a difference in baseline frequencies.
    In this code we average over all given baseline channels. 

    Parameters
    ----------
    :files: List[Path]
        List containing all Path objects for successfully processed files.
    :paired: bool
        For now we are testing on paired data, but this will not always be the case. 
    :save: bool
        Option to save the made baseline statistics dataframe. 
    """
    files = list(power_path.glob("*_power.pkl"))
    baselines = {}
    timepoints = set()

    # Averaging over baselines
    for file in files:
        match = re.match(r"VEP(\d+)_(\d+)_power.pkl", file.name)
        if not match:
            print(f"Unexpected filename in folder: {file.name}")
            continue
        patient, timepoint = match.groups()
        patient, timepoint = int(patient), int(timepoint)
        timepoints.add(timepoint)

        df = pd.read_pickle(file)
        channels = [c for c in df.columns if c.endswith("_BASE")]
        df['Baseline_avg'] = df[channels].mean(axis=1)
        df = df[["Frequency", "Harmonic", "Baseline_avg"]].copy()

        if patient not in baselines:
            baselines[patient] = {}
        baselines[patient][timepoint] = df

    # Defining stimulation & harmonic pairs
    freqs = set()
    for p_data in baselines.values():
        for tp, df in p_data.items():
            pairs = list(zip(df["Frequency"], df["Harmonic"]))
            freqs.update(pairs)
    freqs = sorted(freqs)

    # Remapping data to frequency pairs
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
                    else:
                        print(f"No baseline value for frequency-pair {sel} for patient {patient}.")
                else:
                    print(f"Timepoint {tp} not found for patient {patient}.")
                    continue
            values_per_tp.append(np.array(values))

        # Choosing and performing statistical test
        test, stat, p = None, None, None
        n = len(timepoints)
        if paired:
            if n == 2:
                stat, p = wilcoxon(*values_per_tp)
                test = "Wilcoxon signed-rank"
            elif n > 2:
                stat, p = friedmanchisquare(*values_per_tp)
                test = "Friedman chi-squared"
            else:
                print(f"Can't compare only one baseline, error in {tp}.")
                continue
        else:
            if n == 2:
                stat, p = mannwhitneyu(*values_per_tp)
                test = "Mann-Whitney U"
            elif n > 2:
                stat, p = kruskal(*values_per_tp)
                test = "Kruskal Wallis"
            else:
                print(f"Can't compare only one baseline, error in {tp}.")
                continue

        results.append({"Frequency": freq,
                "Harmonic": harm,
                "Test": test,
                "paired": paired,
                "statistic": stat,
                "Critical chi-squared": chi2.isf(q=0.05, df=n-1) if n > 2 else "Non-applicable",
                "p_value": p})

    df_results = pd.DataFrame(results)

    # Optie om de dataframe op te slaan.
    if save:
        df_results.to_csv("base_stats.csv", sep=";", index=False)
        print("Baseline statistics dataframe saved as base_stats.csv")

    return df_results

def stats_power(power_path: Path, paired: bool = True, save: bool = True, abs: bool = True):
    """
    This code checks if there is a statistical difference between the different timepoints.

    Parameters
    ----------
    :files: List[Path]
        List containing all Path objects for successfully processed files.
    :paired: bool
        For now we are testing on paired data, but this will not always be the case. 
    :save: bool
        Option to save the made baseline statistics dataframe.
    :abs: bool
        Option to either use absolute or relative values. 
    """
    