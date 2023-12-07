import re
import logging

import uproot
import numpy as np
import awkward as ak
from sklearn.model_selection import train_test_split


from cfg.hnl_mva_tools import read_json_file

# ┌───────────────┐
# │ LOGGING LEVEL │
# └───────────────┘
logging.basicConfig(level=logging.INFO)


def read_files_and_open_trees(ntuples_json: str, vars_json: str):
    """
    This function reads the ntuples_json file which contains the file names
    for signal and background ntuples, the treename, and the weight name.

    Labels for signal and background are extracted from the file names 
    and will look something like "mN1p0_ctau10" for signal and 
    "QCD_Pt-80To120_MuEnriched" for background.

    Args:
        ntuples_json: path to the ntuples json file
        vars_json: path to the variables json file
    Returns:
        sig_trees: list of signal trees
        bkg_trees: list of background trees
        good_vars: list of variables to use
        weight_name: name of the weight branch
        sig_labels: list of signal labels
        bkg_labels: list of background labels
    """
    ntuples = read_json_file(ntuples_json)
    ########AIUTO
    signal_file_names = ntuples["signal"]
    background_file_names = ntuples["background"]
    treename = ntuples["treename"]
    weight_name = ntuples["weight_name"]

    sig_labels = [re.findall(r"(mN1p\d+_ctau\d+)", f)[0] for f in signal_file_names]
    bkg_labels = [
        s[s.rfind("QCD_Pt-") : s.rfind("_MuEnriched")] for s in background_file_names
    ]

    # LIST OF VARIABLES TO USE
    good_vars = read_json_file(vars_json)["vars"]
    if "C_pass_gen_matching" in good_vars:
        good_vars.remove("C_pass_gen_matching")

    sig_trees = [uproot.open(f)[treename] for f in signal_file_names]
    bkg_trees = [uproot.open(f)[treename] for f in background_file_names]

    return sig_trees, bkg_trees, good_vars, weight_name, sig_labels, bkg_labels


def filter_trees(trees, tree_labels, mass_list, ctau_list):
    """
    This function takes a list of trees and a list of labels and returns
    only the trees and labels couples whose label contains the mass and
    ctau values in the mass_list and ctau_list.

    Args:
        trees: list of trees
        tree_labels: list of labels for the trees
        mass_list: list of masses to keep
        ctau_list: list of ctaus to keep
    Returns:
        return_trees: list of trees to keep
        return_labels: list of labels to keep
    
    Example:
        Inputs will look like this:
        trees = [tree1, tree2, tree3, tree4]
        tree_labels = ["mN1p0_ctau100", "mN1p0_ctau10", 
                        "mN1p0_ctau1000", "mN1p5_ctau100"]
        mass_list = ["mN1p0"]
        ctau_list = ["ctau100", "ctau10"]
        The output will be:
        return_trees = [tree1, tree2]
        return_labels = ["mN1p0_ctau100", "mN1p0_ctau10"]
    """
    assert isinstance(mass_list[0], str)
    assert isinstance(ctau_list[0], str)

    return_trees = []
    return_labels = []
    for tree, label in zip(trees, tree_labels):
        mn = re.findall(r"(mN\dp\d+)", label)[0]
        ct = re.findall(r"(ctau\d+)", label)[0]
        if mn in mass_list and ct in ctau_list:
            return_trees.append(tree)
            return_labels.append(label)

    return return_trees, return_labels


def get_data(
    sig_tree,
    bkg_trees,
    good_vars,
    weight_name,
    test_fraction,
    rng_seed: int,
    equalnumevents: bool = True,
):
    """
    This function takes the trees and variables and returns the data in the
    form of numpy arrays x, y, w, where x is the data, y is the labels and w
    is the weights. The data is split into training and testing sets.

    Args:
        sig_tree: the signal tree
        bkg_trees: the background trees
        good_vars: the variables to use
        weight_name: the name of the weight branch
        test_fraction: the fraction of data to use for testing
        rng_seed: the seed for the random number generator
        equalnumevents: if True, the signal and background events are
            renormalized so that they have the same number of events
            of the signal
    Returns:
        x_train: the training data
        x_test: the testing data
        y_train: the training labels
        y_test: the testing labels
        w_train: the training weights
        w_test: the testing weights
    """
    logging.info(f"get_data called with equalnumevents = {equalnumevents}")
    # Get data from trees, in form of dictionaries
    data_sig = uproot.concatenate(sig_tree, expressions=good_vars, how=dict)
    data_bkg = uproot.concatenate(bkg_trees, expressions=good_vars, how=dict)
    data_sig_weight = uproot.concatenate(sig_tree, expressions=weight_name, how=dict)
    data_bkg_weight = uproot.concatenate(bkg_trees, expressions=weight_name, how=dict)

    # Convert dictionaries to arrays, transposition is needed to get the right shape
    # e.g. x_sig[0] is the first event and has length len(good_vars)
    x_sig = np.array([ak.flatten(data_sig[var]) for var in good_vars]).T
    x_bkg = np.array([ak.flatten(data_bkg[var]) for var in good_vars]).T

    background_weights = data_bkg_weight[weight_name]

    # weights are single valued, we must broadcast them to match the shape of the data
    # use data_bkg[good_vars[0]] as ak.broadcast_arrays to match shapes
    b_w = ak.flatten(ak.broadcast_arrays(background_weights, data_bkg[good_vars[0]])[0])

    # There should be no NaN values now, commented out for now
    # # Find the rows with NaN values in any array
    # nan_rows_sig = np.isnan(x_sig).any(axis=1)
    # nan_rows_bkg = np.isnan(x_bkg).any(axis=1) | np.isnan(b_w)

    # # Remove the rows with NaN values from all arrays
    # # (we must do this here because we cannot reweight NaN values)
    # x_sig = x_sig[~nan_rows_sig]
    # x_bkg = x_bkg[~nan_rows_bkg]
    # b_w = b_w[~nan_rows_bkg]

    # WEIGHTS
    num_sig = x_sig.shape[0]
    num_bkg = x_bkg.shape[0]
    b_w = np.array(b_w)
    s_w = np.ones(num_sig)

    logging.info(f"number of signal events = {num_sig}")
    logging.info(f"number of background events = {num_bkg}")
    logging.info(f"before renormalization, sum of sig weights = {np.sum(s_w)}")
    logging.info(f"before renormalization, sum of bkg weights = {np.sum(b_w)}")

    # Create full x,y,w arrays

    # REWEIGHTING FOR EQUALNUMEVENTS
    # convert s_w and b_w to numpy arrs
    if equalnumevents:
        renorm_signal = num_sig / np.sum(s_w)
        renorm_background = num_sig / np.sum(b_w)
        s_w *= renorm_signal
        b_w *= renorm_background
        logging.info(f"renormalizing signal weights by {renorm_signal}")
        logging.info(f"renormalizing background weights by {renorm_background}")
        logging.info(f"after renormalization, sum of sig weights = {np.sum(s_w)}")
        logging.info(f"after renormalization, sum of bkg weights = {np.sum(b_w)}")

    x = np.vstack((x_sig, x_bkg))
    y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])
    w = np.hstack([s_w, b_w])

    x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
        x, y, w, test_size=test_fraction, random_state=rng_seed
    )
    return x_train, x_test, y_train, y_test, w_train, w_test
