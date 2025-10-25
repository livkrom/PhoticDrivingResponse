"""
Analytics
"""
import re
import os
import glob
from pathlib import Path
from scipy.stats import wilcoxon, friedmanchisquare, mannwhitneyu, kruskal, chi2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
import seaborn as sns


def stats_base_power(folder_power: str, paired: bool = True, save: bool = False):
    """
    Checks if there is a difference in baseline frequencies.
    In this code we average over all given baseline channels. 

    Parameters
    ----------
    :folder_power: str
        Name of the patient file: VEPxx_T, xx = patient ID, T = timepoint.
    :files: List[Path]
        List containing all Path objects for successfully processed files.
    :paired: bool
        For now we are testing on paired data, but this will not always be the case. 
    :save: bool
        Option to save the made baseline statistics dataframe. 
    """
    path_power = Path(f"./{folder_power}")
    files = list(path_power.glob("*_power.pkl"))
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
        channels = [c for c in df.columns if c.endswith("_BASE")and not c.startswith("Average")]
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

def stats_power(responder_IDs: list, folder_power: str, paired: bool = True, save: bool = False, plot: bool = True):
    """
    This code checks if there is a statistical difference between the different timepoints.

    Parameters
    ----------
    :responders_numbers:
        List of patient_IDs corresponding to patients that respond to the medication. 
    :folder_power: str
        Name of the patient file: VEPxx_T, xx = patient ID, T = timepoint.
    :paired: bool
        For now we are testing on paired data, but this will not always be the case. 
    :save: bool
        Option to save the made baseline statistics dataframe.
    :abs: bool
        Option to either use absolute or relative values. 
    """

    if plot:
        path_power = Path(f"./{folder_power}")
        responders = [f"VEP{idx.zfill(2)}" if not idx.startswith("VEP") else idx for idx in responder_IDs]
        time_map = {"1": "t0", "2": "t1", "3": "t2"}
        df_patient = []

        files = glob.glob(os.path.join(path_power, "*.pkl"))
        for file in files:
            name = os.path.basename(file).replace(".pkl", "")
            parts = name.split("_")
            patient = parts[0]
            time = parts[1]

            data = pd.read_pickle(file)

            data["Patient"] = patient
            data["Time"] = time_map.get(time)
            data["Group"] = "Responder" if patient in responders else "Non-responder"
            data["Absolute Power"] = data["Average_SNR"]

            df_patient.append(data)
        df = pd.concat(df_patient, ignore_index=True)
        df["FreqPair"] = df["Harmonic"].astype(str) + r"$\mathregular{ Hz_{S" + df["Frequency"].astype(str) + "}}$"

        time_order = ["t0", "t1", "t2", "base"]
        full_palette = dict(zip(time_order, sns.color_palette("Set3", len(time_order))))
        present_times = [t for t in time_order if t in df["Time"].unique()]
        filtered_palette = {k: full_palette[k] for k in present_times}

        sns.set(style="whitegrid")
        g = sns.catplot(data=df, x="FreqPair", y="Absolute Power",
                        hue="Time", hue_order=present_times, col="Group", kind='box',
                        palette=filtered_palette, linewidth=0.8, width=0.55,
                        fliersize=3, sharey=True, col_order=["Responder", "Non-responder"],
                        height=6, aspect=1.3, legend=False)
        #g.set(ylim=(df["Absolute Power"][df["Absolute Power"] > 0].min()*0.8, None))
        g.set(ylim=(-10,None))
        for ax in g.axes.flatten():
            xticks = ax.get_xticks()
            xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
            stim_freqs = []

            for label in xticklabels:
                match = re.search(r"S(\d+)", label)
                stim_freqs.append(match.group(1) if match else "unknown")

            change_indices = [i for i in range(1, len(stim_freqs)) if stim_freqs[i] != stim_freqs[i-1]]
            change_indices = [0] + change_indices + [len(stim_freqs)]

            for i in range(len(change_indices) - 1):
                start = xticks[change_indices[i]]
                end = xticks[change_indices[i+1] - 1]
                center = (start + end) / 2
                stim_label = f"S: {stim_freqs[change_indices[i]]} Hz"
                ax.text(center, 1.05, stim_label, ha='center', va='bottom',
                        fontsize=10, transform=ax.get_xaxis_transform())
                if i < len(change_indices) - 2:
                    ax.axvline(x=end + 0.5, color='gray', linestyle='--', alpha=0.3)

        g.set_axis_labels("Analyzed Frequency (Hz) - Stimulation Frequency (Hz)", "Absolute Power (dB µV^2)")
        for ax in g.axes.flatten():
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.tick_params(axis='x', rotation=45)
        group_counts = df.groupby("Group")["Patient"].nunique()

        # Manually set titles per facet
        for ax, group in zip(g.axes.flatten(), g.col_names):
            count = group_counts.get(group, 0)
            ax.set_title(f"{group} (n = {count})", fontsize=13)

        handles = [Patch(facecolor=filtered_palette[t], label=t) for t in present_times]
        for ax in g.axes.flatten():
            ax.legend(
                handles=handles, title="Condition", loc="upper right",
                frameon=True, facecolor="white", edgecolor="gray", fontsize=9
            ).get_frame().set_alpha(0.9)

        g.fig.suptitle("Absolute Power distributions by dose condition and response group", fontsize=15, y=1)
        g.fig.subplots_adjust(top=0.90, right=0.97, wspace=0.15)
        plt.show()

    if save:
        df.to_csv("Powers_Abs_All.csv", index=True)

def stats_plv(responder_ids: list, folder_plv: str, paired: bool = True, save: bool = True, plot: bool = True):
    """
    This code checks if there is a statistical difference between the different timepoints.

    Parameters
    ----------
    :responders_numbers:
        List of patient_IDs corresponding to patients that respond to the medication. 
    :folder_power: str
        Name of the patient file: VEPxx_T, xx = patient ID, T = timepoint.
    :paired: bool
        For now we are testing on paired data, but this will not always be the case. 
    :save: bool
        Option to save the made baseline statistics dataframe.
    :abs: bool
        Option to either use absolute or relative values. 
    """

    if plot:
        path_plv = Path(f"./{folder_plv}")
        responders = [f"VEP{idx.zfill(2)}" if not idx.startswith("VEP") else idx for idx in responder_ids]
        time_map = {"1": "t0", "2": "t1", "3": "t2"}
        df_patient = []

        files = glob.glob(os.path.join(path_plv, "*.pkl"))

        for file in files:
            name = os.path.basename(file).replace(".pkl", "")
            parts = name.split("_")
            patient = parts[0]
            time = parts[1]

            stim_file = os.path.join(path_plv, f"{patient}_{time}_plv_stim.pkl")
            base_file = os.path.join(path_plv, f"{patient}_{time}_plv_base.pkl")

            data_stim = pd.read_pickle(stim_file)
            data_base = pd.read_pickle(base_file)

            for df, condition in [(data_stim, time_map.get(time)), (data_base, "base")]:
                df = df.copy()
                df["Patient"] = patient
                df["Time"] = condition
                df["Group"] = "Responder" if patient in responders else "Non-responder"
                df["PLV"] = df["mean_plv"]
                df["FreqPair"] = df["Harmonic"].astype(str) + r"$\mathregular{ Hz_{"+ "S" + df["Frequency"].astype(str)+"}}$"
                df["StimFreq"] = df["Frequency"].astype(str) + " Hz"
                df_patient.append(df)

        df = pd.concat(df_patient, ignore_index=True)

        time_order = ["t0", "t1", "t2", "base"]
        full_palette = dict(zip(time_order, sns.color_palette("Set3", len(time_order))))
        present_times = [t for t in time_order if t in df["Time"].unique()]
        filtered_palette = {k: full_palette[k] for k in present_times}

        sns.set(style="whitegrid")
        g = sns.catplot(data=df, x="FreqPair", y="PLV",
                        hue="Time", col="Group", kind='box', hue_order=present_times,
                        palette=filtered_palette, linewidth=0.8, width=0.55,
                        fliersize=3, sharey=True, col_order=["Responder", "Non-responder"],
                        height=6, aspect=1.3, legend=False)

        g.set_axis_labels("Analyzed Frequency (Hz) - Stimulation Frequency (Hz)", "Phase Locking Value (PLV)")
        g.set(ylim=(0,1))

        for ax in g.axes.flatten():
            xticks = ax.get_xticks()
            xticklabels = [tick.get_text() for tick in ax.get_xticklabels()]
            stim_freqs = []

            for label in xticklabels:
                match = re.search(r"S(\d+)", label)
                stim_freqs.append(match.group(1) if match else "unknown")

            change_indices = [i for i in range(1, len(stim_freqs)) if stim_freqs[i] != stim_freqs[i-1]]
            change_indices = [0] + change_indices + [len(stim_freqs)]

            for i in range(len(change_indices) - 1):
                start = xticks[change_indices[i]]
                end = xticks[change_indices[i+1] - 1]
                center = (start + end) / 2
                stim_label = f"S: {stim_freqs[change_indices[i]]} Hz"
                ax.text(center, 1.05, stim_label, ha='center', va='bottom',
                        fontsize=10, transform=ax.get_xaxis_transform())
                if i < len(change_indices) - 2:
                    ax.axvline(x=end + 0.5, color='gray', linestyle='--', alpha=0.3)

        for ax in g.axes.flatten():
            ax.grid(axis='y', linestyle='--', alpha=0.4)
            ax.tick_params(axis='x', rotation=45)

        group_counts = df.groupby("Group")["Patient"].nunique()
        for ax, group in zip(g.axes.flatten(), g.col_names):
            count = group_counts.get(group, 0)
            ax.set_title(f"{group} (n = {count})", fontsize=13)

        handles = [Patch(facecolor=filtered_palette[t], label=t) for t in present_times]
        for ax in g.axes.flatten():
            ax.legend(
                handles=handles, title="Condition", loc="upper right",
                frameon=True, facecolor="white", edgecolor="gray", fontsize=9
            ).get_frame().set_alpha(0.9)

        g.fig.suptitle("PLV distributions by dose condition and response group", fontsize=15, y=1)
        g.fig.subplots_adjust(top=0.90, right=0.97, wspace=0.15)
        plt.show()
        plt.tight_layout(pad=2)

        if save:
            df.to_csv("PLV_Stats_All.csv", index=False)
