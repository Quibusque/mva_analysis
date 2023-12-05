import uproot
import re
import awkward as ak
import numpy as np

from cfg.hnl_mva_tools import read_json_file
from data_tools.load_data import filter_trees

if __name__ == "__main__":
    json_file = read_json_file("source/cfg/ntuples.json")
    signal_files = json_file["signal"]
    new_signal_files = json_file["signal_new"]
    background_files = json_file["background"]
    treename = json_file["treename"]
    weight_name = json_file["weight_name"]

    #list variables
    good_vars = read_json_file("source/cfg/vars_new.json")["vars"]
    #add weight and event (normally not included in variables because they are not variables to train on)
    good_vars.append(weight_name)
    good_vars.append("event")


    #you can choose my_signal_files and my_new_signal_files if you want to
    #modify only some specific files
    my_signal_files = signal_files
    my_new_signal_files = new_signal_files
    for signal_file,new_signal_file in zip(my_signal_files,my_new_signal_files):
        #open the root file and tree with uproot
        file = uproot.open(signal_file)
        tree = file[treename]

        #load the variables you want to use
        events = tree.arrays(good_vars, library="ak", how=dict)

        new_events = {}

        for key,arr in events.items():
            #skip the C_pass_gen_matching variable, it must not be added to the new tree
            if key == "C_pass_gen_matching":
                continue
            new_events[key] = []
            #distiction between iterable element or not is done
            #to preserve structure of original ttree
            if isinstance(arr[0],ak.highlevel.Array):
                for my_arr,C_pass in zip(arr,events["C_pass_gen_matching"]):
                    for element,must_pass in zip(my_arr,C_pass):
                        if must_pass:
                            new_events[key].append(ak.Array([element]))
            else:
                for element,must_pass in zip(arr,events["C_pass_gen_matching"]):
                    if any(must_pass):
                        new_events[key].append(element)
            #convert to np array
            new_events[key] = ak.Array(new_events[key])

            #this is a crosscheck for valid_events
            valid_evts = 0
            for arr in events["C_pass_gen_matching"]:
                for element in arr:
                    if element == 1:
                        valid_evts+=1

            for key,arr in new_events.items():
                if len(arr) != valid_evts:
                    print(f"ERROR: {key} has {len(arr)} elements, while there should be {valid_evts}")
                    break
                for element in arr:
                    if isinstance(element,ak.highlevel.Array):
                        if len(element) != 1:
                            print(f"ERROR: {key} has {len(element)} elements, while there should be 1")
                            break

        #Now that I have the new correct version of the events dictionary, save it to a TTree
        #and then save the TTree to a root file

        with uproot.recreate(new_signal_file) as f:
            f[treename] = new_events
            print(f"Saved new TTree to {new_signal_file}")