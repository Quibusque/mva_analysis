import uproot
import re
import awkward as ak
import numpy as np
import argparse

from cfg.hnl_mva_tools import read_json_file
from data_tools.load_data import filter_trees


def categorize(events_dict):
    Sig_arr = events_dict["C_Hnl_vertex_2DSig_BS"]
    Hnl_charge_arr = events_dict["C_mu_Hnl_charge"]
    Ds_charge_arr = events_dict["C_mu_Ds_charge"]
    n_evts = len(Sig_arr)
    category_arr = [np.zeros(len(Sig_arr[i])) for i in range(n_evts)]
    for i in range(n_evts):
        n_sub_evts = len(Sig_arr[i])
        for j in range(n_sub_evts):
            # SAME SIGN
            if Hnl_charge_arr[i][j] == Ds_charge_arr[i][j]:
                if Sig_arr[i][j] < 50:
                    category_arr[i][j] = int(1)
                elif Sig_arr[i][j] < 150:
                    category_arr[i][j] = int(2)
                else:
                    category_arr[i][j] = int(3)
            # OPPOSITE SIGN
            else:
                if Sig_arr[i][j] < 50:
                    category_arr[i][j] = int(4)
                elif Sig_arr[i][j] < 150:
                    category_arr[i][j] = int(5)
                else:
                    category_arr[i][j] = int(6)

    events_dict["C_category"] = ak.Array(category_arr)
    return events_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        action="store_true",
        help="categorize the signal and background trees",
    )
    args = parser.parse_args()

    json_file = read_json_file("source/cfg/ntuples.json")
    signal_files = json_file["signal"]
    new_signal_files = json_file["signal_new"]
    background_files = json_file["background"]
    new_background_files = json_file["background_new"]
    treename = json_file["treename"]
    weight_name = json_file["weight_name"]

    # list variables
    good_vars = read_json_file("source/cfg/vars_new.json")["vars"]
    # add weight and event (normally not included in variables because they are not variables to train on)
    good_vars.append(weight_name)

    # ┌─────────────────────────────────┐
    # │ GEN MATCH SIGNAL AND CATEGORIZE │
    # └─────────────────────────────────┘
    for signal_file, new_signal_file in zip(signal_files, new_signal_files):
        # open the root file and tree with uproot
        file = uproot.open(signal_file)
        tree = file[treename]

        # load the variables you want to use
        events = tree.arrays(good_vars, library="ak", how=dict)

        if args.category:
            events = categorize(events)

        new_events = {}

        for key, arr in events.items():
            myPrint = False
            # skip the C_pass_gen_matching variable, it must not be added to the new tree
            if key == "C_pass_gen_matching":
                continue
            new_events[key] = []
            # print keys of new_events
            if myPrint:
                print(new_events.keys())
            # distiction between iterable element or not is done
            # to preserve structure of original ttree
            if isinstance(arr[0], ak.highlevel.Array):
                for my_arr, C_pass in zip(arr, events["C_pass_gen_matching"]):
                    for element, must_pass in zip(my_arr, C_pass):
                        if myPrint:
                            print(f"loop1 element: {element}, must_pass: {must_pass}")
                        if must_pass:
                            new_events[key].append(np.array([element]))
            else:
                for element, must_pass in zip(arr, events["C_pass_gen_matching"]):
                    if myPrint:
                        print(f"loop 2 element: {element}, must_pass: {must_pass}")
                    if any(must_pass):
                        new_events[key].append(element)
            # convert to np array
            new_events[key] = np.array(new_events[key])

            # this is a crosscheck for valid_events
            valid_evts = 0
            for arr in events["C_pass_gen_matching"]:
                for element in arr:
                    if element == 1:
                        valid_evts += 1

        for key, arr in new_events.items():
            if len(arr) != valid_evts:
                print(
                    f"ERROR: {key} has {len(arr)} elements, while there should be {valid_evts}"
                )
                break
            for element in arr:
                if isinstance(element, ak.highlevel.Array):
                    if len(element) != 1:
                        print(
                            f"ERROR: {key} has {len(element)} elements, while there should be 1"
                        )
                        break

        # Now that I have the new correct version of the events dictionary, save it to a TTree
        # and then save the TTree to a root file

        # remove keys that are not in good_vars and are not C_category
        for key in list(new_events.keys()):
            if key not in good_vars and key != "C_category":
                new_events.pop(key)

        with uproot.recreate(new_signal_file) as f:
            f[treename] = new_events
            print(f"Saved new TTree to {new_signal_file}")

    # ┌───────────────────────┐
    # │ CATEGORIZE BACKGROUND │
    # └───────────────────────┘
    if "C_pass_gen_matching" in good_vars:
        good_vars.remove("C_pass_gen_matching")

    for background_file, new_background_file in zip(
        background_files, new_background_files
    ):
        break
        # open the root file and tree with uproot
        file = uproot.open(background_file)
        tree = file[treename]

        # load the variables you want to use
        events = tree.arrays(good_vars, library="ak", how=dict)

        events = categorize(events)

        # remove keys that are not in good_vars and are not C_category
        for key in list(events.keys()):
            if key not in good_vars and key != "C_category":
                events.pop(key)

        with uproot.recreate(new_background_file) as f:
            f[treename] = events
            print(f"Saved new TTree to {new_background_file}")
