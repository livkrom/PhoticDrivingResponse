"""
Kladkladklad 
Deze module gebruik ik puur om functies te testen.
"""

from pathlib import Path
#import argparse

#import matplotlib
#matplotlib.use("TkAgg")

#from patients import parse_args, patient_files, eeg, filter_files
#from power import Power
from analytics import stats_base

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
# parser.add_argument("-t", "--time", choices = time_map.keys(), default="t0",
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

# # Testing functions
# eeg = eeg(src, passband, notch = 50, plot=False)
# df = Power._stimulation(eeg, save=False)
# epochs, df_epochs = Power._epoch(df, eeg, save=False, plot=False)
# fft_powers, fft_freqs, epochs = Power._fft_blocks(passband, epochs, df_epochs, t
# rim=0.0, padding= "copy", occi=True, plot=False)
# powers = Power._snr(passband, epochs, fft_powers, fft_freqs, save=True, plot=True, 
# harms=4, montage="standard_1020")

## Filtering files function

# === MOCK / LOAD TRIAL MAP AND ARGS ===
# Adjust to match your actual trial_map and args structure
time_map = {
    "t0": ["t0", "t1", "t2"],
    "t1": ["t0", "t1"],
    "t2": ["t0", "t1"]
}
class Args:
    time = "all"
    trial = "t0"
args = Args()


# # === FILTER FILES ===
# complete = filter_files(list(Path("./results_POWER").glob("*_power.pkl")), time_map, args)
# print(f"Complete sets of files for: {complete}")

# === Analytics ===
power_path = Path("./results_POWER")
stats_base(power_path, paired=True, save=True)
