"""
This is the main module that calls all made functions.
"""
from pathlib import Path
from patients import parse_args, patient_files, eeg
from power import Power

if __name__ == "__main__":
    trial_map, args = parse_args()
    pt_files = patient_files(trial_map, args)
    skipped = []

    passband = [0.5, 100]
    for pt_file in pt_files:
        print(f"Processing {pt_file.name}...")

        try:
            snr = Power(passband, eeg(pt_file, passband)).run()

            save_path = Path("./results_SNR") / f"{pt_file.stem}_snr.pkl"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            snr.to_pickle(save_path)
            print(f"Saved SNR to {save_path}")

        except Exception as e: # pylint: disable=broad-except
            print(f"Skipping {pt_file.name} due to error: {e}")
            skipped.append((pt_file.name, str(e)))
            continue

    print("Folder analysis completed.")
    if skipped:
        print("Skipped patients:")
        for name, reason in skipped:
            print(f"{name}: {reason}")
