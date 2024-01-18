import numpy as np
import os
import uproot
import awkward as ak
import ROOT
from cfg.hnl_mva_tools import read_json_file

from data_tools.load_data import (
    get_categorized_data,
    categorize_data,
    read_files_and_open_trees
)
from mva_tools.mva_training_tools import (
    train_one_signal_all_methods,
    load_model,
)

training_vars = ["C_Ds_pt"]
category_list = [1, 2, 3, 4, 5, 6]
category_var = "C_category"
tree_name = "final_tree"

def fix_pred_shape(y_pred, mask):
    """
    This function takes the y_pred 1D array and reshapes it to match the true
    entries of the mask array. False entries are filled with np.nan.
    Args:
        y_pred: the array of predictions
        mask: the array of mask values
    Returns:
        y_pred_shaped: the reshaped array of predictions
    Example:
        y_pred = [1,2,3]
        mask = [[False, True], [True, False, True]]
        y_pred_reshaped = [[nan, 1],[2, nan, 3]]
    """
    assert len(y_pred) == ak.sum(mask)

    y_pred_flat = np.full(ak.sum(ak.num(mask)), np.nan)
    y_pred_flat[ak.flatten(mask)] = y_pred

    y_pred_shaped = ak.unflatten(y_pred_flat, ak.num(mask))
    return y_pred_shaped

def get_bdt_output(data_dict, training_vars, category_list, xgboost_models):
    assert len(category_list) == len(xgboost_models)

    # categorize the data, this adds the C_category column
    categorize_data(
        data_dict,
        category_list,
        category_var=category_var,
        default_category=0,
    )

    # make sure the default category is empty
    assert np.sum(data_dict[category_var] == 0) == 0
    # input_dict[var] is an awkward array
    # make out an awkward array that copies
    # the shape of input_dict[var]
    bdt_output = ak.ones_like(data_dict[training_vars[0]]) * np.nan

    for category,model in zip(category_list,xgboost_models):
        mask = data_dict[category_var] == category
        # x_cat[i] is ith event
        # x_cat[i][j] is the jth variable for the ith event
        x_cat = np.array([ak.flatten(data_dict[var][mask]) for var in training_vars]).T
        y_score_cat = model.predict(x_cat, output_margin=True)
        y_score_cat_shaped = fix_pred_shape(y_score_cat, mask)

        #ak.where(condition, x, y) does the same as
        # output[i] = x[i] if condition[i] else y[i]     so this fills the 
        #bdt_output array with the predictions for the current category and
        #leaves the rest as they are
        bdt_output = ak.where(mask, y_score_cat_shaped, bdt_output)

    #check that there are no np.nan values left in the bdt_output array
    assert np.sum(np.isnan(bdt_output)) == 0

    #check again that bdt_output has the same shape as the input data
    assert ak.all(ak.num(bdt_output) == ak.num(data_dict[training_vars[0]]))

    return bdt_output


def rewrite_root_file(input_file, tree_name, bdt_output):  # array_of_pNN2):
    print("Save new branch in original ROOT file")
    myfile = ROOT.TFile(input_file, "update")
    mytree = myfile.Get(tree_name)
    bdt_curr_score = ak.Array([0.5])
    new_branch = mytree.Branch("C_bdtscore", bdt_curr_score, "C_bdtscore[nCand]/D")
    n_events = mytree.GetEntries()
    assert len(bdt_output) == n_events
    for i in range(n_events):
        bdt_curr_score = bdt_output[i]
        mytree.GetEntry(i)
        new_branch.Fill()


def write_output_to_root_signal(tree, file, mass):
    # print ("[signal] Reading trees {} in file {}".format(tree, file))
    print(f"[signal] Reading trees {tree} in file {file}")
    data = uproot.open(file)[tree]

    out = data.arrays(training_vars, library="ak", how="dict")
    out["type"] = 1.0

    mass = float(mass)

    bdt_score = get_pNN_output_signal(
        out, mass, cat2_model="both_cat-2/pnn-preproc-drop-l2_fullrun2_2"
    )
    # pNN_out_array2 = get_pNN_output_signal(out, mass, cat2_model='both_cat-2/pnn-preproc-drop-l2_v4_second')

    # update root file with output pNN
    rewrite_root_signal(file, tree, bdt_score)  # , pNN_out_array2)
