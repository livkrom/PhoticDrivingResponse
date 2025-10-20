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

parser = argparse.ArgumentParser()
trial_map = {
    "t0": "T0_T1_T2",
    "t1": "T0_T1",
    "t2": "T0_T1"}
parser.add_argument("-tr", "--trial", choices = trial_map.keys(), default="t0",
help = "Choose trial map: t0, t1 or t2")

time_map = {
    "t0": ["t0", "t1", "t2"],
    "t1": ["t0", "t1"],
    "t2": ["t0", "t1"]
}
parser.add_argument("-t", "--time", choices = time_map.keys(), default="t2",
                    help = "Choose point in time: t0, t1 or t2")

while True:
    args = parser.parse_args()
    if args.time not in time_map[args.trial]:
        print(f"Error: timepoint {args.time} is not in trial {args.trial}")
        print("Please choose again.")
        continue
    break

data_map = {
    "t0": "1",
    "t1": "2",
    "t2": "3"
}

pt = "VEP38_" + data_map[args.time] + ".cnt"
src = Path('/Volumes/Docs/Bruikbare Data') / trial_map[args.trial] / args.time / pt
passband = [0.5, 100]

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
raw = eeg(src, passband, notch = 50, plot=False)
df = Phase._stimulation_phase(raw, save=False, base=False)
epochs =  Phase._epoch_phase(df, raw)
phases = Phase._fft_phase(epochs, occi=True, plot = False, save=True)

# Baseline phase functions
# df, baseline_blocks = Phase._stimulation_phase(raw, save=True, base=True)
# epochs_baseline =  Phase._epoch_phase(baseline_blocks, raw)
# phases_baseline = Phase._fft_phase(epochs_baseline, occi=True, plot=True, save=False)


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
