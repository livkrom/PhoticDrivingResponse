"""
This is the main module that calls all made functions.
"""
from patients import parse_args, patient_files, eeg, save_pickle_results, filter_files
from power import Power
from phase import Phase
from analytics import stats_base_power, stats_power, stats_plv

if __name__ == "__main__":
    # Reading data
    trial_map, time_map, args = parse_args()
    pt_files = patient_files(trial_map, args)
    skipped, complete_power, complete_plv = [], [], []

    passband = [0.5, 100]
    all_mean_plv = {}
    folder_power = "results_POWER"
    folder_plv = "results_PLV"
    n = 0

    # for pt_file in pt_files:
    #     n += 1
    #     print(f"Processing file {n} out of {len(pt_files)}.")

    #     # Power calculation
    #     try:
    #         raw = eeg(pt_file, passband, occi=True, plot=False)
    #         power = Power(passband, raw).run()
    #         save_pickle_results(power, pt_file, folder_power, feat="power")

    #     except Exception as e: # pylint: disable=broad-except
    #         print(f"Skipping power calculation of {pt_file.name} due to error: {e}")
    #         skipped.append((pt_file.name, "Power", str(e)))
    #         continue

    #     # PLV calculation
    #     try:
    #         plv_stim, plv_base = Phase(passband, raw).run()
    #         save_pickle_results({"stim": plv_stim, "base": plv_base}, pt_file, folder_plv, feat="plv")

    #     except Exception as e:
    #         print(f"Skipping phase calculation of {pt_file.name} due to Phase error: {e}")
    #         skipped.append((pt_file.name, "PLV", str(e)))
    #         continue

    complete_power = filter_files(folder_power, time_map, args, feat="power")
    complete_plv = filter_files(folder_plv, time_map, args, feat="plv")

    if skipped:
        print(f"Files processed. Skipped files: {skipped}")
        print(f"Complete power files for {complete_power}")
        print(f"Complete plv files for {complete_plv}")
    else:
        print("All files processed.")
        print(f"Complete power files for {complete_power}")
        print(f"Complete plv files for {complete_plv}")

    # Statistics
    responder_ids = {"2", "10", "11", "17", "21", "22", "32", "40", "46", "48", "51", "57", "63"}
    # stats_base_power(folder_power, paired=True, save=True) # Power baseline
    stats_power(responder_ids, folder_power, paired=True, save=True, plot=True)
    stats_plv(responder_ids, folder_plv, paired=True, save=True, plot=True)
