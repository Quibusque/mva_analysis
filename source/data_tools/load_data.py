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


def myprintdict(dict):
    """
    This function prints the content of a dictionary in a nice way i.e.
    key1: [value1, value2, ...] (total values: number_of_values)
    key2: [value1, value2, ...] (total values: number_of_values)
    ...
    last_key: [value1, value2, ...] (total values: number_of_values)
    total_keys: number_of_keys
    """
    for key in dict.keys():
        print(f"{key}: {dict[key][:2]}  (total values: {len(dict[key])})")
    print(f"total keys: {len(dict.keys())}")


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

    sig_trees = [uproot.open(f)[treename] for f in signal_file_names]
    bkg_trees = [uproot.open(f)[treename] for f in background_file_names]

    return sig_trees, bkg_trees, weight_name, sig_labels, bkg_labels


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


def categorize_data(
    data,
    category_list,
    category_var: str,
    default_category=0,
):
    conditions = [
        (data["C_Hnl_vertex_2DSig_BS"] < 50)
        & (data["C_mu_Hnl_charge"] == data["C_mu_Ds_charge"]),
        ((data["C_Hnl_vertex_2DSig_BS"] > 50) & (data["C_Hnl_vertex_2DSig_BS"] < 150))
        & (data["C_mu_Hnl_charge"] == data["C_mu_Ds_charge"]),
        (data["C_Hnl_vertex_2DSig_BS"] > 150)
        & (data["C_mu_Hnl_charge"] == data["C_mu_Ds_charge"]),
        (data["C_Hnl_vertex_2DSig_BS"] < 50)
        & (data["C_mu_Hnl_charge"] != data["C_mu_Ds_charge"]),
        ((data["C_Hnl_vertex_2DSig_BS"] > 50) & (data["C_Hnl_vertex_2DSig_BS"] < 150))
        & (data["C_mu_Hnl_charge"] != data["C_mu_Ds_charge"]),
        (data["C_Hnl_vertex_2DSig_BS"] > 150)
        & (data["C_mu_Hnl_charge"] != data["C_mu_Ds_charge"]),
    ]

    # Use ak.where to select categories based on conditions
    data[category_var] = ak.where(
        conditions[0],
        category_list[0],
        ak.where(
            conditions[1],
            category_list[1],
            ak.where(
                conditions[2],
                category_list[2],
                ak.where(
                    conditions[3],
                    category_list[3],
                    ak.where(
                        conditions[4],
                        category_list[4],
                        ak.where(conditions[5], category_list[5], default_category),
                    ),
                ),
            ),
        ),
    )


    return data


def gen_match_data(data, gen_matching_var: str):
    mask = data[gen_matching_var] == 1
    masked_data = {key: ak.mask(data[key], mask) for key in data}
    
    # Drop None values from each array in masked_data
    masked_data = {key: ak.drop_none(value) for key, value in masked_data.items()}
    
    for key in masked_data.keys():
        masked_data[key] = masked_data[key][ak.count(masked_data[key], axis=-1) > 0]
    
    return masked_data


def get_categorized_data(
    sig_tree,
    bkg_trees,
    good_vars,
    training_vars,
    weight_name,
    test_fraction,
    rng_seed: int,
    validation_fraction: float = None,
    gen_matching_var: str = "C_pass_gen_matching",
    category_list=[1, 2,3, 4, 5, 6],
    equalnumevents: bool = True,
    scale_factor_vars = None,
    category_var = "C_category",
):
    logging.info(f"get_categorized_data called with equalnumevents = {equalnumevents}")
    assert category_var not in good_vars

    data = []

    additional_vars = ["C_Hnl_vertex_2DSig_BS", "C_mu_Hnl_charge", "C_mu_Ds_charge"]

    # Get data from trees, in form of dictionaries
    data_sig = uproot.concatenate(
        sig_tree,
        expressions=good_vars
        + scale_factor_vars
        + additional_vars
        + [gen_matching_var],
        how=dict,
    )
    data_bkg = uproot.concatenate(
        bkg_trees, expressions=good_vars + scale_factor_vars + additional_vars, how=dict
    )
    data_bkg_weight = uproot.concatenate(bkg_trees, expressions=weight_name, how=dict)

    data_sig = gen_match_data(data_sig, gen_matching_var)

    data_sig = categorize_data(
        data_sig,
        category_list,
        category_var,
        default_category=0,
    )
    data_bkg = categorize_data(
        data_bkg, category_list, category_var, default_category=0
    )

    # check that the default_category is not used
    print(
        f"number of signal events in default category = {np.sum(data_sig[category_var]==0)}"
    )
    print(
        f"number of background events in default category = {np.sum(data_bkg[category_var]==0)}"
    )

    # pop away the additional_vars that are not in good_vars
    # because they must not be used in the training
    for var in additional_vars:
        if var not in good_vars:
            data_sig.pop(var)
            data_bkg.pop(var)
    data_sig.pop(gen_matching_var)
    # find the index of the category_var in data_sig.keys() and data_bkg.keys()
    # this is needed later to remove the category_var column
    C_cat_index_sig = list(data_sig.keys()).index(category_var)
    C_cat_index_bkg = list(data_bkg.keys()).index(category_var)
    print(f"C_cat_index_sig = {C_cat_index_sig}")
    print(f"C_cat_index_bkg = {C_cat_index_bkg}")
    print(f"length of arrays in data_sig = {[len(data_sig[key]) for key in data_sig.keys()]}")
    print(f"length of arrays in data_bkg = {[len(data_bkg[key]) for key in data_bkg.keys()]}")
    #take first length and find vars with different length
    first_length_sig = len(data_sig[list(data_sig.keys())[0]])
    first_length_bkg = len(data_bkg[list(data_bkg.keys())[0]])
    for key in data_sig.keys():
        if len(data_sig[key]) != first_length_sig:
            print(f"length of {key} = {len(data_sig[key])}")
    for key in data_bkg.keys():
        if len(data_bkg[key]) != first_length_bkg:
            print(f"length of {key} = {len(data_bkg[key])}")

    for category in category_list:
        # build x_sig and x_bkg for the category, i.e. data[category_var]==category

        mask_sig = data_sig[category_var] == category
        mask_bkg = data_bkg[category_var] == category

        x_sig = np.array(
            [
                ak.flatten(data_sig[var][mask_sig])
                for var in training_vars
            ]
        ).T
        x_bkg = np.array(
            [
                ak.flatten(data_bkg[var][mask_bkg])
                for var in training_vars
            ]
        ).T

        # ┌─────────────────────┐
        # │ BUILD SCALE FACTORS │
        # └─────────────────────┘
        sig_sf = np.array(
            [
                ak.flatten(
                    data_sig[scale_factor_vars[0]][data_sig[category_var] == category]
                )
            ]
        ).T
        for scale_factor_var in scale_factor_vars[1:]:
            sig_sf *= np.array(
                [
                    ak.flatten(
                        data_sig[scale_factor_var][data_sig[category_var] == category]
                    )
                ]
            ).T
        bkg_sf = np.array(
            [
                ak.flatten(
                    data_bkg[scale_factor_vars[0]][data_bkg[category_var] == category]
                )
            ]
        ).T
        for scale_factor_var in scale_factor_vars[1:]:
            bkg_sf *= np.array(
                [
                    ak.flatten(
                        data_bkg[scale_factor_var][data_bkg[category_var] == category]
                    )
                ]
            ).T

        # flatten sig_sf and bkg_sf
        sig_sf = np.array(sig_sf).flatten()
        bkg_sf = np.array(bkg_sf).flatten()

        background_weights = data_bkg_weight[weight_name]

        # weights are single valued, we must broadcast them to match the shape of the data
        # use data_bkg[good_vars[0]] as ak.broadcast_arrays to match shapes
        broadcast_bkg_weights = ak.broadcast_arrays(
            background_weights, data_bkg[good_vars[0]]
        )[0]
        b_w = ak.flatten(broadcast_bkg_weights[data_bkg[category_var] == category])

        # WEIGHTS
        num_sig = x_sig.shape[0]
        num_bkg = x_bkg.shape[0]
        b_w = np.array(b_w)
        s_w = np.ones(num_sig)

        # ADD SCALE FACTORS TO WEIGHTS
        s_w *= sig_sf
        b_w *= bkg_sf

        logging.info(f"category {category}: shape of x_sig = {x_sig.shape}")
        logging.info(f"category {category}: shape of x_bkg = {x_bkg.shape}")
        logging.info(
            f"category {category}: before renormalization, sum of sig weights = {np.sum(s_w)}"
        )
        logging.info(
            f"category {category}: before renormalization, sum of bkg weights = {np.sum(b_w)}"
        )

        # Create full x,y,w arrays

        # REWEIGHTING FOR EQUALNUMEVENTS
        # convert s_w and b_w to numpy arrs
        if equalnumevents:
            renorm_signal = num_sig / np.sum(s_w)
            renorm_background = num_sig / np.sum(b_w)
            s_w *= renorm_signal
            b_w *= renorm_background
            logging.info(
                f"category {category}: renormalizing signal weights by {renorm_signal}"
            )
            logging.info(
                f"category {category}: renormalizing background weights by {renorm_background}"
            )
            logging.info(
                f"category {category}: after renormalization, sum of sig weights = {np.sum(s_w)}"
            )
            logging.info(
                f"category {category}: after renormalization, sum of bkg weights = {np.sum(b_w)}"
            )

        x = np.vstack((x_sig, x_bkg))
        y = np.hstack([np.ones(num_sig), np.zeros(num_bkg)])
        w = np.hstack([s_w, b_w])
        if validation_fraction is None:
            x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
                x, y, w, test_size=test_fraction, random_state=rng_seed
            )
            data.append((x_train, x_test, y_train, y_test, w_train, w_test))
        else:
            x_train, x_test, y_train, y_test, w_train, w_test = train_test_split(
                x, y, w, test_size=test_fraction, random_state=rng_seed
            )
            # the apparently weird test_size is to make percentages work properly
            # e.g. if test=0.2 and validation=0.1,
            # then the validation set is 0.1/(1-0.2)=0.125 of the training set
            x_train, x_val, y_train, y_val, w_train, w_val = train_test_split(
                x_train,
                y_train,
                w_train,
                test_size=validation_fraction / (1 - test_fraction),
                random_state=rng_seed,
            )
            data.append(
                (x_train, x_val, x_test, y_train, y_val, y_test, w_train, w_val, w_test)
            )

    return data


def get_categorized_test_data(tree,training_vars,
    category_list=[1, 2,3, 4, 5, 6],
    category_var = "C_category"
):
    logging.info(f"get_categorized_test_data called with categories = {category_list}")
    assert category_var not in training_vars

    data_arr = []

    additional_vars = ["C_Hnl_vertex_2DSig_BS", "C_mu_Hnl_charge", "C_mu_Ds_charge"]

    # Get data from trees, in form of dictionaries
    data = uproot.concatenate(
        tree,
        expressions=training_vars
        + additional_vars,
        how=dict,
    )

    print(f"type of data['C_Ds_pt']= {type(data['C_Ds_pt'])}")
    print(f"ak.type(data['C_Ds_pt'])= {ak.type(data['C_Ds_pt'])}")
    data = categorize_data(
        data,
        category_list,
        category_var,
        default_category=0,
    )

    # check that the default_category is not used
    print(
        f"number of events in default category = {np.sum(data[category_var]==0)}"
    )

    # pop away the additional_vars that are not in good_vars
    # because they must not be used in the training
    for var in additional_vars:
        if var not in training_vars:
            data.pop(var)
    # find the index of the category_var in data.keys()
    # this is needed later to remove the category_var column
    C_cat_index = list(data.keys()).index(category_var)
    print(f"C_cat_index = {C_cat_index}")
    print(f"length of arrays in data = {[len(data[key]) for key in data.keys()]}")
    #take first length and find vars with different length
    first_length = len(data[list(data.keys())[0]])
    for key in data.keys():
        if len(data[key]) != first_length:
            print(f"length of {key} = {len(data[key])}")

    for category in category_list:
        # build x_sig and x_bkg for the category, i.e. data[category_var]==category

        mask = data[category_var] == category

        print(f"mask length = {len(mask)}, mask = {mask}")

        x = np.array(
            [
                ak.flatten(data[var][mask])
                for var in training_vars
            ]
        ).T
        num_evts = x.shape[0]

        logging.info(f"category {category}: shape of x = {x.shape}")

        # Create full x,y


        y = np.zeros(num_evts)
        data_arr.append((x, y))

    return data_arr
