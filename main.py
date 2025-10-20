"""
This is the main module that calls all made functions.
"""
from pathlib import Path
import pandas as pd
from patients import parse_args, patient_files, eeg, filter_files
from power import Power
from phase import Phase
from analytics import stats_base

if __name__ == "__main__":
    # Reading data
    trial_map, time_map, args = parse_args()
    pt_files = patient_files(trial_map, args)
    skipped, complete = [], []

    passband = [0.5, 100]
    all_mean_plv = {}
    for pt_file in pt_files:
        print(f"Processing {pt_file.name}...")
        raw = eeg(pt_file, passband)

        # # Power calculation
        # try: 
        #     power_path = Path("./results_POWER")
        #     power = Power(passband, raw).run()
        #     power_path_singles = power_path / f"{pt_file.stem}_power.pkl"
        #     power_path_singles.parent.mkdir(parents=True, exist_ok=True)
        #     power.to_pickle(power_path_singles)
        #     complete.append(pt_file)
        #     print(f"Saved powers to {power_path_singles}")
        # except Exception as e: # pylint: disable=broad-except
        #     print(f"Skipping power calculation of {pt_file.name} due to error: {e}")
        #     skipped.append((pt_file.name, "Power", str(e)))
        #     continue
        
        # PLV calculation
        try:
            phase_path = Path("./results_PLV")
            phases = Phase(passband, raw).run()
            patient_plv = {k: v["mean_plv"] for k, v in phases.items()}
            all_mean_plv[pt_file.stem] = pd.Series(patient_plv)
        except Exception as e:
            print(f"Skipping phase calculation of {pt_file.name} due to Phase error: {e}")
            skipped.append((pt_file.name, "Phase", str(e)))
            continue

    all_mean_plv.to_csv('PLV.csv', index=False)
    print(f"Saved PLV to {phase_path}")

    print("Folder analysis completed.")
    if skipped:
        print("Skipped patients:")
        for name, function, reason in skipped:
            print(f"{name}: {function}, {reason}")

    # # File filtering power
    # complete_power = filter_files(list(power_path.glob("*_power.pkl")), time_map, args)
    # print(f"In power calculation, complete sets of files for: {complete_power}")

    # # Statistics
    # stats_base(power_path, paired=True, save=True) # Power
