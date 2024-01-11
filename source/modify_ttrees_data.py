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
        "--input",
        type=str,
        help="Input data file",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output data file",
    )
    args = parser.parse_args()

    old_file = args.input
    new_file = args.output

    json_file = read_json_file("source/cfg/ntuples.json")
    treename = json_file["treename"]

    # list variables
    good_vars = read_json_file("source/cfg/vars_new.json")["training_vars"]


    # open the root file and tree with uproot
    file = uproot.open(old_file)
    tree = file[treename]

    # load the variables you want to use
    events = tree.arrays(good_vars, library="ak", how=dict)

    events = categorize(events)

    # remove keys that are not in good_vars and are not C_category
    for key in list(events.keys()):
        if key not in good_vars and key != "C_category":
            events.pop(key)

    with uproot.recreate(new_file) as f:
        f[treename] = events
        print(f"Saved new TTree to {new_file}")
