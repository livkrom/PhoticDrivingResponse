"""
This is the main module that calls all made functions.
"""
from pathlib import Path
import pandas as pd
from patients import parse_args, patient_files, eeg, save_pickle_results, filter_files
from power import Power
from phase import Phase
from analytics import stats_base_power, stats_power

if __name__ == "__main__":
    # Reading data
    trial_map, time_map, args = parse_args()
    pt_files = patient_files(trial_map, args)
    skipped, complete_power, complete_plv = [], [], []

    passband = [0.5, 100]
    all_mean_plv = {}
    folder_power = "results_POWER"
    folder_plv = "results_PLV"

    for pt_file in pt_files:
        print(f"Processing {pt_file.name}...")

        # Power calculation
        try: 
            raw = eeg(pt_file, passband, occi=True, plot=False)
            power = Power(passband, raw).run()
            save_pickle_results(power, pt_file, folder_power, complete_power, feat="power", verbose=True)

        except Exception as e: # pylint: disable=broad-except
            print(f"Skipping power calculation of {pt_file.name} due to error: {e}")
            skipped.append((pt_file.name, "Power", str(e)))
            continue
        
        # PLV calculation
        try:
            raw = eeg(pt_file, passband, occi=True, plot=False)
            plv_stim, plv_base = Phase(passband, raw).run()
            save_pickle_results([plv_stim, plv_base], pt_file, folder_plv, complete_plv, feat="plv", verbose=True)

        except Exception as e:
            print(f"Skipping phase calculation of {pt_file.name} due to Phase error: {e}")
            skipped.append((pt_file.name, "PLV", str(e)))
            continue

    print("Folder analysis completed.")
    if skipped:
        print("Skipped patients:")
        for name, function, reason in skipped:
            print(f"{name}: {function}, {reason}")

    # # File filtering power
    # complete_power = filter_files(list(power_path.glob("*_power.pkl")), time_map, args)
    # print(f"In power calculation, complete sets of files for: {complete_power}")

    # Statistics
    responder_ids = {"2", "10", "11", "17", "21", "22", "32", "40", "46", "48", "51", "57", "63"}
    stats_base_power(folder_power, paired=True, save=True) # Power baseline
    stats_power(responder_ids, folder_power,paired=True, save=False, plot=True)
    stats_phase(responder_ids, folder_plv, )

