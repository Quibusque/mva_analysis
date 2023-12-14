import json, re, os
import numpy as np
import pandas as pd
import argparse
import logging

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from mva_tools.mva_training_tools import (
    train_one_signal_all_methods,
    train_one_signal_all_methods_categorized,
    load_model,
    methods_list,
)
from data_tools.load_data import (
    read_files_and_open_trees,
    get_data,
    get_categorized_data,
    filter_trees,
)
from mva_tools.mva_response_tools import (
    my_predict,
    model_response_hists,
    save_results_to_csv,
)
from mva_tools.mva_plot_tools import plot_corr_matrix
from my_logging import log_weights, log_histo_weights, log_num_events

methods_list = ["keras_shallow", "adaboost", "XGBoost"]


if __name__ == "__main__":
    # parser args
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="train models")
    parser.add_argument("--results", action="store_true", help="produce results")
    parser.add_argument(
        "--mass", type=str, help="single mass label of form mN1p0", required=True
    )
    parser.add_argument("--out_dir", type=str, help="output directory", required=True)

    args = parser.parse_args()

    ntuples_json = "source/cfg/ntuples.json"
    vars_json = "source/cfg/vars_new.json"

    out_dir = args.out_dir
    out_dir = f"{out_dir}/myMVA"

    # open trees from root files
    (
        sig_trees,
        bkg_trees,
        good_vars,
        weight_name,
        sig_labels,
        bkg_labels,
    ) = read_files_and_open_trees(ntuples_json, vars_json, additional_vars=["C_category"])
    print(f"file reading is done")
    # ┌────────────────────────────────┐
    # │ CHOOSE YOUR MASSES TO TRAIN ON │
    # └────────────────────────────────┘
    mass_list = [args.mass]
    my_sig_trees, my_sig_labels = filter_trees(
        sig_trees, sig_labels, mass_list=mass_list, ctau_list=["ctau1000"]
    )

    # ┌───────────────┐
    # │ TEST FRACTION │
    # └───────────────┘
    test_fraction = 0.2

    # leonardo cut-based efficiencies
    cut_based_dict = {
        "mN1p8_ctau10": {"sig_eff": 0.629, "bkg_eff": 0.021},
        "mN1p8_ctau100": {"sig_eff": 0.693, "bkg_eff": 0.021},
        "mN1p8_ctau1000": {"sig_eff": 0.6, "bkg_eff": 0.021},
        "mN1p5_ctau10": {"sig_eff": 0.679, "bkg_eff": 0.021},
        "mN1p5_ctau100": {"sig_eff": 0.688, "bkg_eff": 0.021},
        "mN1p5_ctau1000": {"sig_eff": 0.653, "bkg_eff": 0.021},
        "mN1p25_ctau10": {"sig_eff": 0.624, "bkg_eff": 0.021},
        "mN1p25_ctau100": {"sig_eff": 0.628, "bkg_eff": 0.021},
        "mN1p25_ctau1000": {"sig_eff": 0.654, "bkg_eff": 0.021},
        "mN1p0_ctau10": {"sig_eff": 0.547, "bkg_eff": 0.021},
        "mN1p0_ctau100": {"sig_eff": 0.59, "bkg_eff": 0.021},
        "mN1p0_ctau1000": {"sig_eff": 0.538, "bkg_eff": 0.021},
    }

    for key, value in cut_based_dict.items():
        sig_eff = value["sig_eff"]
        bkg_eff = value["bkg_eff"]
        value["purity"] = sig_eff / (sig_eff + bkg_eff)

    my_cut_based_dict = {
        key: value for key, value in cut_based_dict.items() if key in my_sig_labels
    }

    # ┌─────────────────────────────────────────┐
    # │ TRAINING (IF ARGUMENT --train IS GIVEN) │
    # └─────────────────────────────────────────┘
    if args.train:
        for sig_tree, sig_label in zip(my_sig_trees, my_sig_labels):
            print(f"Training {sig_label} ...")
            results_dir = f"{out_dir}/{sig_label}"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            # GET DATA, CATEGORIZED
            categories=[1, 2, 4, 5, 6]
            full_data = get_categorized_data(
                sig_tree,
                bkg_trees,
                good_vars,
                weight_name,
                test_fraction,
                rng_seed=42,
                equalnumevents=True,
                category_list=categories,
            )
            # TRAINING
            for category,data in zip(categories,full_data):
                print(f"Training category {category} ...")
                x_train, x_test, y_train, y_test, w_train, w_test = data
                log_weights(y_train, y_test, w_train, w_test, sig_label)
                log_num_events(y_train, y_test, sig_label)
                # SIGNAL AND BACKGROUND TRAINING AND TEST SAMPLES
                x_train_sig = x_train[y_train == 1]
                x_train_bkg = x_train[y_train == 0]
                x_test_sig = x_test[y_test == 1]
                x_test_bkg = x_test[y_test == 0]
                # corresponding weights
                w_train_sig = w_train[y_train == 1]
                w_train_bkg = w_train[y_train == 0]
                w_test_sig = w_test[y_test == 1]
                w_test_bkg = w_test[y_test == 0]

                #PLOT CORRELATION MATRIX
                plot_corr_matrix(x_train_sig, x_test_sig, good_vars, results_dir,f"sig_cat{category}")
                plot_corr_matrix(x_train_bkg, x_test_bkg, good_vars, results_dir,f"bkg_cat{category}")

                # TRAIN
                train_one_signal_all_methods_categorized(
                    x_train, y_train, w_train, methods_list, results_dir, new_vars=True, category_index=category
                )
                print(f"Training category {category} complete!")
            print(f"Training {sig_label} complete!")