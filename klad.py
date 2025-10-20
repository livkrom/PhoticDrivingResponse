"""
Kladkladklad 
Deze module gebruik ik puur om functies te testen.
"""

from pathlib import Path
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

import mne
from mne.io.base import BaseRaw

from patients import parse_args, patient_files, eeg, filter_files
# from power import Power
from phase import Phase
#from analytics import stats_base

# parser = argparse.ArgumentParser()
# trial_map = {
#     "t0": "T0_T1_T2",
#     "t1": "T0_T1",
#     "t2": "T0_T1"}
# parser.add_argument("-tr", "--trial", choices = trial_map.keys(), default="t0",
# help = "Choose trial map: t0, t1 or t2")

# time_map = {
#     "t0": ["t0", "t1", "t2"],
#     "t1": ["t0", "t1"],
#     "t2": ["t0", "t1"]
# }
# parser.add_argument("-t", "--time", choices = time_map.keys(), default="t2",
#                     help = "Choose point in time: t0, t1 or t2")

# while True:
#     args = parser.parse_args()
#     if args.time not in time_map[args.trial]:
#         print(f"Error: timepoint {args.time} is not in trial {args.trial}")
#         print("Please choose again.")
#         continue
#     break

# data_map = {
#     "t0": "1",
#     "t1": "2",
#     "t2": "3"
# }

# pt = "VEP38_" + data_map[args.time] + ".cnt"
# src = Path('/Volumes/Docs/Bruikbare Data') / trial_map[args.trial] / args.time / pt
# passband = [0.5, 100]

# # Comparing EEG without max filter
# eeg = eeg(src, passband, notch = 50, plot=True)
# raw = mne.io.read_raw_ant(src, preload=True, verbose='ERROR')
# raw.filter(l_freq=0.05, h_freq=1999, picks="eeg", verbose='ERROR')
# line_freq = 50 if (freq := raw.info["line_freq"]) is None else freq
# lowpass = np.arange(line_freq, raw.info["lowpass"]+1, line_freq)
# raw.notch_filter(freqs=(lowpass), notch_widths=(lowpass)/line_freq, picks=["eeg"], verbose='ERROR')
# raw.plot(scalings = "auto", title="Non-Filtered EEG data", show=True, block=False)

# plt.show(block=True)
# Testing power functions
# eeg = eeg(src, passband, notch = 50, plot=True)
# df = Power._stimulation_power(eeg, save=False)
# epochs, df_epochs = Power._epoch(df, eeg, save=False, plot=True)
# fft_powers, fft_freqs, epochs = Power._fft_blocks(passband, epochs, df_epochs, t
# rim=0.0, padding= "copy", occi=True, plot=False)
# powers = Power._snr(passband, epochs, fft_powers, fft_freqs, save=True, plot=True,
# harms=4, montage="standard_1020")

# Testing phase functions
# raw = eeg(src, passband, notch = 50, plot=False)
# df = Phase._stimulation_phase(raw, save=False, base=False)
# epochs =  Phase._epoch_phase(df, raw)
# phases = Phase._fft_phase(epochs, occi=True, plot = False, save=True)

# Baseline phase functions
# raw = eeg(src, passband, notch = 50, occi=True, plot=False)
# df, df_base = Phase._stimulation_phase(raw, save=False, base=False)
# epochs_baseline =  Phase._epoch_phase(df_base, raw)
# phases_baseline = Phase._fft_phase(epochs_baseline, occi=True, plot=False, save=False)


## Filtering files function

# # === MOCK / LOAD TRIAL MAP AND ARGS ===
# # Adjust to match your actual trial_map and args structure
# time_map = {
#     "t0": ["t0", "t1", "t2"],
#     "t1": ["t0", "t1"],
#     "t2": ["t0", "t1"]
# }
# class Args:
#     """Beepboop"""
#     time = "all"
#     trial = "t0"
# args = Args()


# # # === FILTER FILES ===
# # complete = filter_files(list(Path("./results_POWER").glob("*_power.pkl")), time_map, args)
# # print(f"Complete sets of files for: {complete}")

# # === Analytics ===
# power_path = Path("./results_POWER")
# stats_base(power_path, paired=True, save=True)

# # # === PHASE RESULTS AAhH ===
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re

# ----- SETTINGS -----
order = [
    (6, 6), (6, 12), (6, 18), (6, 24), (6, 30), (6, 36),
    (10, 10), (10, 20), (10, 30), (10, 40),
    (20, 20), (20, 40)
]
freq_labels = [f"{f1}-{f2}" for f1, f2 in order]

# ----- LOAD -----
plv_stim = pd.read_csv("PLV_stim.csv")
plv_base = pd.read_csv("PLV_base.csv")

# ---- Ensure structure ----
if plv_stim.shape[0] != len(order):
    raise ValueError(f"Expected 12 rows (frequency pairs), got {plv_stim.shape[0]}.")

# ---- Extract condition (t0, t1, t2) from column names ----
def extract_condition(name):
    match = re.search(r"_(\d)$", name)
    if match:
        idx = match.group(1)
        return {"1": "t0 (no meds)", "2": "t1 (1 unit)", "3": "t2 (max dose)"}.get(idx, "unknown")
    return "unknown"

# Melt stimulation data (long format)
stim_long = plv_stim.melt(var_name="Patient_Time", value_name="PLV")
stim_long["FreqPair"] = freq_labels * (plv_stim.shape[1])
stim_long["Condition"] = stim_long["Patient_Time"].apply(extract_condition)
stim_long["Patient"] = stim_long["Patient_Time"].apply(lambda x: re.sub(r"_\d$", "", x))

# Melt baseline data (no time condition)
base_long = plv_base.melt(var_name="Patient_Time", value_name="PLV")
base_long["FreqPair"] = freq_labels * (plv_base.shape[1])
base_long["Condition"] = "Baseline"
base_long["Patient"] = base_long["Patient_Time"]

# Combine both datasets
df_all = pd.concat([stim_long, base_long], ignore_index=True)

# ---- Filter valid conditions ----
condition_order = ["Baseline", "t0 (no meds)", "t1 (1 unit)", "t2 (max dose)"]
palette = {
    "Baseline": "royalblue",
    "t0 (no meds)": "firebrick",
    "t1 (1 unit)": "darkorange",
    "t2 (max dose)": "mediumorchid"
}

# ----- PLOTTING -----
plt.figure(figsize=(16, 7))
sns.boxplot(
    data=df_all,
    x="FreqPair", y="PLV", hue="Condition",
    hue_order=condition_order,
    palette=palette,
    linewidth=0.8,
    width=0.6,          # thinner boxes for more spacing between freq pairs
    fliersize=3
)
plt.ylim(0, 1)
plt.xlabel("Frequency pair (Hz)")
plt.ylabel("Phase Locking Value (PLV)")
plt.title("PLV distributions across dose conditions and baseline")
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(title="", frameon=True)
plt.tight_layout()
plt.show()
