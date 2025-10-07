"""
This is the main module that calls all made functions.
"""
from pathlib import Path
from patients import parse_args, patient_files, eeg, filter_files
from power import Power
from analytics import stats_base

if __name__ == "__main__":
    # Reading data
    trial_map, time_map, args = parse_args()
    pt_files = patient_files(trial_map, args)
    skipped, complete = [], []

    passband = [0.5, 100]
    for pt_file in pt_files:
        print(f"Processing {pt_file.name}...")

        try: # Power calculation
            power_path = Path("./results_POWER")
            power = Power(passband, eeg(pt_file, passband)).run()
            power_path_singles = power_path / f"{pt_file.stem}_power.pkl"
            power_path_singles.parent.mkdir(parents=True, exist_ok=True)
            power.to_pickle(power_path_singles)
            complete.append(pt_file)
            print(f"Saved powers to {power_path_singles}")

        except Exception as e: # pylint: disable=broad-except
            print(f"Skipping {pt_file.name} due to error: {e}")
            skipped.append((pt_file.name, str(e)))
            continue

    print("Folder analysis completed.")
    if skipped:
        print("Skipped patients:")
        for name, reason in skipped:
            print(f"{name}: {reason}")

    # File filtering
    complete = filter_files(list(power_path.glob("*_power.pkl")), time_map, args)
    print(f"Complete sets of files for: {complete}")

    # Statistics
    stats_base(power_path, paired=True, save=True)
