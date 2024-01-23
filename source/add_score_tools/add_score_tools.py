import numpy as np
import os
import uproot
import awkward as ak
import ROOT
from array import array
from cfg.hnl_mva_tools import read_json_file

from data_tools.load_data import (
    get_categorized_data,
    categorize_data,
    read_files_and_open_trees,
)
from mva_tools.mva_training_tools import (
    train_one_signal_all_methods,
    load_model,
)

training_vars = ["C_Ds_pt"]
category_list = [1, 2, 3, 4, 5, 6]
category_var = "C_category"

def print_ascii_histogram(data, num_bins=10):

    if ak.Array(data).ndim != 1:
        flat_data = np.array(ak.flatten(data))
    else:
        flat_data = np.array(data)

    # Calculate histogram
    hist, bin_edges = np.histogram(flat_data, bins=num_bins)

    # Find maximum count to normalize bar heights
    max_count = max(hist)

    # For each bin, print an ASCII bar
    for count, edge in zip(hist, bin_edges):
        # Normalize bar height
        height = int(np.ceil((count / max_count) * 50))
        # Use a fixed width of 7 for the bin edge values
        print(f'{edge:7.2f} | ' + '#' * height)

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

    for category, model in zip(category_list, xgboost_models):
        mask = data_dict[category_var] == category
        # x_cat[i] is ith event
        # x_cat[i][j] is the jth variable for the ith event
        x_cat = np.array([ak.flatten(data_dict[var][mask]) for var in training_vars]).T
        y_score_cat = model.predict(x_cat, output_margin=True)
        y_score_cat_shaped = fix_pred_shape(y_score_cat, mask)

        # ak.where(condition, x, y) does the same as
        # output[i] = x[i] if condition[i] else y[i]     so this fills the
        # bdt_output array with the predictions for the current category and
        # leaves the rest as they are
        bdt_output = ak.where(mask, y_score_cat_shaped, bdt_output)

    # check that there are no np.nan values left in the bdt_output array
    assert np.sum(np.isnan(bdt_output)) == 0

    # check again that bdt_output has the same shape as the input data
    assert ak.all(ak.num(bdt_output) == ak.num(data_dict[training_vars[0]]))

    return bdt_output

def get_multiple_bdt_outputs(data_dict, training_vars, category_list, xgboost_models_dict):

    mass_hypotheses = list(xgboost_models_dict.keys())
    assert len(category_list) == len(xgboost_models_dict[mass_hypotheses[0]])

    # categorize the data, this adds the C_category column
    categorize_data(
        data_dict,
        category_list,
        category_var=category_var,
        default_category=0,
    )

    # make sure the default category is empty
    assert np.sum(data_dict[category_var] == 0) == 0
    bdt_outputs = []
    for mass_hyp in mass_hypotheses:
        # input_dict[var] is an awkward array
        # make out an awkward array that copies
        # the shape of input_dict[var]
        bdt_output = ak.ones_like(data_dict[training_vars[0]]) * np.nan

        for category, model in zip(category_list, xgboost_models_dict[mass_hyp]):
            mask = data_dict[category_var] == category
            # x_cat[i] is ith event
            # x_cat[i][j] is the jth variable for the ith event
            x_cat = np.array([ak.flatten(data_dict[var][mask]) for var in training_vars]).T
            y_score_cat = model.predict(x_cat, output_margin=True)
            y_score_cat_shaped = fix_pred_shape(y_score_cat, mask)

            # ak.where(condition, x, y) does the same as
            # output[i] = x[i] if condition[i] else y[i]     so this fills the
            # bdt_output array with the predictions for the current category and
            # leaves the rest as they are
            bdt_output = ak.where(mask, y_score_cat_shaped, bdt_output)

        # check that there are no np.nan values left in the bdt_output array
        assert np.sum(np.isnan(bdt_output)) == 0

        # check again that bdt_output has the same shape as the input data
        assert ak.all(ak.num(bdt_output) == ak.num(data_dict[training_vars[0]]))


        bdt_outputs.append(bdt_output)
    print(f"bdt_outputs: {bdt_outputs}")
    return bdt_outputs



# THIS VERSION ADDS A NEW BRANCH CALLED nCand_presel
# AND FILLS IT WITH THE NUMBER OF CANDIDATES PER EVENT
# AND FILLS THE BDT SCORES IN THE C_Bdt_score BRANCH


def append_score_root(input_file_name, output_file_name, treename, bdt_score, score_name):
    # use uproot to read the tree
    # just to check consistency quickly

    max_subevts = 0

    with uproot.open(input_file_name) as f:
        tree = f[treename]
        # check matching number of entries
        # and subentries
        var = list(tree.keys())[0]
        var_array = tree[var].array(library="ak")
        assert ak.all(ak.num(var_array) == ak.num(bdt_score))
        max_subevts = np.nanmax(np.array(ak.num(var_array)))

    # now use ROOT TFile to append the score to the tree
    # and save to a new file

    # Open the input file
    input_file = ROOT.TFile.Open(input_file_name)
    # Get the tree
    tree = input_file.Get(treename)
    # Create a new file
    output_file = ROOT.TFile.Open(output_file_name, "RECREATE")
    # Clone the tree
    output_tree = tree.CloneTree(0)  # Don't copy the entries yet

    # check total number of events
    assert len(bdt_score) == tree.GetEntries()

    # Create a new branch

    bdt_score_branch = array("d", [0] * max_subevts)
    nCand_presel = array("i", [0])
    output_tree.Branch("nCand_presel", nCand_presel, "nCand_presel/I")
    output_tree.Branch(score_name, bdt_score_branch, f"{score_name}[nCand_presel]/D")
    # output_tree.Branch("C_Bdt_score", bdt_score_branch, "C_Bdt_score[nCand_presel]/D")

    # Now loop over the tree and fill the new branch
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)  # Load the i-th entry
        nCand_presel[0] = len(bdt_score[i])  # Set the number of candidates
        for j in range(nCand_presel[0]):
            bdt_score_branch[j] = bdt_score[i][j]  # Set the BDT scores
        output_tree.Fill()  # Fill the new tree

    # Write the new tree to the output file
    output_tree.Write()

    # Close the files
    input_file.Close()
    output_file.Close()

def append_score_root_overwrite(input_file_name,treename,bdt_score,score_name):
    # use uproot to read the tree
    # just to check consistency quickly

    max_subevts = 0

    with uproot.open(input_file_name) as f:
        tree = f[treename]
        # check matching number of entries
        # and subentries
        var = list(tree.keys())[0]
        var_array = tree[var].array(library="ak")
        assert ak.all(ak.num(var_array) == ak.num(bdt_score))
        max_subevts = np.nanmax(np.array(ak.num(var_array)))

    # now use ROOT TFile to append the score to the tree
    # and save to a new file

    # Open the input file
    input_file = ROOT.TFile.Open(input_file_name,"UPDATE")
    # Get the tree
    tree = input_file.Get(treename)
    output_tree = tree.CloneTree(0)  # Don't copy the entries yet

    # check total number of events
    assert len(bdt_score) == tree.GetEntries()

    # Create a new branch

    bdt_score_branch = array("d", [0] * max_subevts)
    nCand_presel = array("i", [0])
    output_tree.Branch("nCand_presel", nCand_presel, "nCand_presel/I")
    output_tree.Branch(score_name, bdt_score_branch, f"{score_name}[nCand_presel]/D")
    # output_tree.Branch("C_Bdt_score", bdt_score_branch, "C_Bdt_score[nCand_presel]/D")

    # Now loop over the tree and fill the new branch
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)  # Load the i-th entry
        nCand_presel[0] = len(bdt_score[i])  # Set the number of candidates
        for j in range(nCand_presel[0]):
            bdt_score_branch[j] = bdt_score[i][j]  # Set the BDT scores
        output_tree.Fill()  # Fill the new tree

    # Write the new tree
    output_tree.Write("",ROOT.TFile.kOverwrite)
    input_file.Close()

def append_multiple_scores_overwrite(input_file_name, treename, bdt_score_list, score_name_list):
    # use uproot to read the tree
    # just to check consistency quickly

    max_subevts = 0

    with uproot.open(input_file_name) as f:
        tree = f[treename]
        # check matching number of entries
        # and subentries
        var = list(tree.keys())[0]
        var_array = tree[var].array(library="ak")
        assert ak.all(ak.num(var_array) == ak.num(bdt_score_list[0]))
        max_subevts = np.nanmax(np.array(ak.num(var_array)))

    # now use ROOT TFile to append the score to the tree
    # and save to a new file

    # Open the input file
    input_file = ROOT.TFile.Open(input_file_name,"UPDATE")
    # Get the tree
    tree = input_file.Get(treename)
    output_tree = tree.CloneTree(0)  # Don't copy the entries yet

    # check total number of events
    assert len(bdt_score_list[0]) == tree.GetEntries()

    # Create a new branch

    nCand_presel = array("i", [0])
    output_tree.Branch("nCand_presel", nCand_presel, "nCand_presel/I")
    print("in append_multiple_scores_overwrite")
    for bdt_score, score_name in zip(bdt_score_list, score_name_list):
        print(f"bdt_score: {bdt_score}")
        bdt_score_branch = array("d", [0] * max_subevts)
        output_tree.Branch(score_name, bdt_score_branch, f"{score_name}[nCand_presel]/D")
        # output_tree.Branch("C_Bdt_score", bdt_score_branch, "C_Bdt_score[nCand_presel]/D")

        # Now loop over the tree and fill the new branch
        for i in range(tree.GetEntries()):
            tree.GetEntry(i)  # Load the i-th entry
            nCand_presel[0] = len(bdt_score[i])  # Set the number of candidates
            for j in range(nCand_presel[0]):
                bdt_score_branch[j] = bdt_score[i][j]  # Set the BDT scores
            output_tree.Fill()  # Fill the new tree

    # Write the new tree
    output_tree.Write("",ROOT.TFile.kOverwrite)
    input_file.Close()
    print(f"Appended {score_name_list} to {input_file_name}")

def write_multiple_scores_root(input_file_name, output_file_name, treename, bdt_score_list, score_name_list):
    # use uproot to read the tree
    # just to check consistency quickly

    max_subevts = 0

    with uproot.open(input_file_name) as f:
        tree = f[treename]
        # check matching number of entries
        # and subentries
        var = list(tree.keys())[0]
        var_array = tree[var].array(library="ak")
        assert ak.all(ak.num(var_array) == ak.num(bdt_score_list[0]))
        max_subevts = np.nanmax(np.array(ak.num(var_array)))

    # now use ROOT TFile to append the score to the tree
    # and save to a new file

    # Open the input file
    input_file = ROOT.TFile.Open(input_file_name)
    # Get the tree
    tree = input_file.Get(treename)
    # Create a new file
    output_file = ROOT.TFile.Open(output_file_name, "RECREATE")
    # Clone the tree
    output_tree = tree.CloneTree(0)  # Don't copy the entries yet

    # check total number of events
    assert len(bdt_score_list[0]) == tree.GetEntries()

    # Create a new branch

    nCand_presel = array("i", [0])
    output_tree.Branch("nCand_presel", nCand_presel, "nCand_presel/I")
    print("in append_multiple_scores_overwrite")
    score_branches = []
    bdt_score_branches = []
    for bdt_score, score_name in zip(bdt_score_list, score_name_list):
        print(f"bdt_score hist: {bdt_score}")
        print_ascii_histogram(bdt_score)
        bdt_score_branches.append(array("d", [0] * max_subevts))
        score_branches.append(output_tree.Branch(score_name, bdt_score_branches[-1], f"{score_name}[nCand_presel]/D"))
        # output_tree.Branch(score_name, bdt_score_branch, f"{score_name}[nCand_presel]/D")
        # output_tree.Branch("C_Bdt_score", bdt_score_branch, "C_Bdt_score[nCand_presel]/D")

    # Now loop over the tree and fill the new branch
    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        nCand_presel[0] = len(bdt_score_list[0][i])
        for j in range(nCand_presel[0]):
            for bdt_score_branch, bdt_score in zip(bdt_score_branches, bdt_score_list):
                bdt_score_branch[j] = bdt_score[i][j]
        output_tree.Fill()

    #close the input file
    input_file.Close()

    # Write the new tree to the output file
    output_tree.Write()

    # Close the output file
    output_file.Close()
    print(f"Wrote {score_name_list} into {output_file_name}")



# # THIS VERSION USES THE ALREADY EXISTING BRANCH nCand
# # BUT IT DOES NOT WORK PERFECTLY BECAUSE nCand IS
# # NOT ALWAYS EQUAL TO THE NUMBER OF CANDIDATES IN THE EVENT
# # BUT IT CAN BE LARGER

# def append_score_root(input_file_name,output_file_name,treename,bdt_score):
#     #use uproot to read the tree
#     #just to check consistency

#     max_subevts = 0

#     with uproot.open(input_file_name) as f:
#         tree = f[treename]
#         #check matching number of entries
#         #and subentries
#         var = list(tree.keys())[0]
#         var_array = tree[var].array(library="ak")
#         assert ak.all(ak.num(var_array) == ak.num(bdt_score))
#         max_subevts = np.nanmax(np.array(ak.num(var_array)))


#     #now use ROOT TFile to append the score to the tree
#     #and save to a new file

#     # Open the input file
#     input_file = ROOT.TFile.Open(input_file_name)
#     # Get the tree
#     tree = input_file.Get(treename)
#     # Create a new file
#     output_file = ROOT.TFile.Open(output_file_name, "RECREATE")
#     # Clone the tree
#     output_tree = tree.CloneTree(0)  # Don't copy the entries yet

#     #check total number of events
#     assert len(bdt_score) == tree.GetEntries()

#     # Create a new branch
#     bdt_score_branch = array('d', [0]*max_subevts)
#     # Get the nCand branch
#     nCand_branch = tree.GetBranch('nCand')
#     output_tree.Branch('C_Bdt_score', bdt_score_branch, 'C_Bdt_score[nCand]/D')

#     # Now loop over the tree and fill the new branch
#     for i in range(tree.GetEntries()):
#         tree.GetEntry(i)  # Load the i-th entry
#         nCand = nCand_branch.GetLeaf('nCand').GetValue()  # Get the value of nCand for this entry
#         for j in range(len(bdt_score[i])):
#             bdt_score_branch[j] = bdt_score[i][j]  # Set the BDT scores
#         output_tree.Fill()  # Fill the new tree

#     # Write the new tree to the output file
#     output_tree.Write()

#     # Close the files
#     input_file.Close()
#     output_file.Close()
