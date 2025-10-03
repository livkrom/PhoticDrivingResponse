"""
Kladkladklad
"""

from pathlib import Path
import argparse

import matplotlib
matplotlib.use("TkAgg")

from patients import eeg
from power import Power

parser = argparse.ArgumentParser()
trial_map = {
    "t0": "T0_T1_T2",
    "t1": "T0_T1",
    "t2": "T0_T1"}
parser.add_argument("-tr", "--trial", choices = trial_map.keys(), default="t0", help = "Choose trial map: t0, t1 or t2")

time_map = {
    "t0": ["t0", "t1", "t2"],
    "t1": ["t0", "t1"],
    "t2": ["t0", "t1"]
}
parser.add_argument("-t", "--time", choices = time_map.keys(), default="t0",
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

# Testing functions
eeg = eeg(src, passband, notch = 50, plot=False)
df = Power._stimulation(eeg, save=False)
epochs, df_epochs = Power._epoch(df, eeg, save=False, plot=False)
fft_powers, fft_freqs = Power._fft_blocks(passband, epochs, df_epochs, trim=0.0, padding= "copy", occi=False, plot=True)
