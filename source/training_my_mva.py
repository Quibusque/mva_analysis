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
    load_model,
    methods_list,
)
from data_tools.load_data import read_files_and_open_trees, get_data, filter_trees
from mva_tools.mva_response_tools import (
    my_predict,
    model_response_hists,
    save_results_to_csv,
)
from mva_tools.mva_plot_tools import plot_corr_matrix
from my_logging import log_weights, log_histo_weights, log_num_events

methods_list = ["keras_shallow"]


if __name__ == "__main__":
    # parser args
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="train models")
    parser.add_argument("--results", action="store_true", help="produce results")
    group2 = parser.add_mutually_exclusive_group(required=True)
    group2.add_argument("--new", action="store_true", help="use new vars")
    group2.add_argument("--old", action="store_true", help="use old vars")
    parser.add_argument(
        "--mass", type=str, help="single mass label of form mN1p0", required=True
    )
    parser.add_argument("--out_dir", type=str, help="output directory", required=True)

    args = parser.parse_args()

    ntuples_json = "source/cfg/ntuples.json"
    if args.old:
        vars_json = "source/cfg/vars_old.json"
    elif args.new:
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
    ) = read_files_and_open_trees(ntuples_json, vars_json)

    # ┌────────────────────────────────┐
    # │ CHOOSE YOUR MASSES TO TRAIN ON │
    # └────────────────────────────────┘
    mass_list = [args.mass]
    my_sig_trees, my_sig_labels = filter_trees(
        sig_trees, sig_labels, mass_list=mass_list, ctau_list=["ctau10"]
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
            # GET DATA x-y arrays
            x_train, x_test, y_train, y_test, w_train, w_test = get_data(
                sig_tree, bkg_trees, good_vars, weight_name, test_fraction, rng_seed=0
            )
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
            plot_corr_matrix(x_train_sig, x_test_sig, good_vars, results_dir,"sig")
            plot_corr_matrix(x_train_bkg, x_test_bkg, good_vars, results_dir,"bkg")

            # TRAIN
            train_one_signal_all_methods(
                x_train, y_train, w_train, methods_list, results_dir, new_vars=args.new
            )
            print(f"Training {sig_label} complete!")

    # ┌─────────────────────────────────────────────────────────────────────────────────────┐
    # │ APPLY MODEL TO TEST SAMPLE AND GET VARIOUS RESULTS (IF ARGUMENT --results IS GIVEN) │
    # └─────────────────────────────────────────────────────────────────────────────────────┘
    if args.results:
        for sig_tree, sig_label in zip(my_sig_trees, my_sig_labels):
            print(f"Producing results for {sig_label} ...")
            results_dir = f"{out_dir}/{sig_label}"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # GET DATA x-y arrays
            x_train, x_test, y_train, y_test, w_train, w_test = get_data(
                sig_tree,
                bkg_trees,
                good_vars,
                weight_name,
                test_fraction,
                rng_seed=42,
                equalnumevents=False,
            )
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

            # print weights total
            logging.info(f"Applying model to test sample ...")
            log_weights(y_train, y_test, w_train, w_test, sig_label)

            for method in methods_list:
                # PREDICT
                model = load_model(f"{results_dir}/{method}_model", method)
                y_train_sig_pred = my_predict(model, x_train_sig)
                y_train_bkg_pred = my_predict(model, x_train_bkg)
                y_test_sig_pred = my_predict(model, x_test_sig)
                y_test_bkg_pred = my_predict(model, x_test_bkg)

                # RESPONSE HISTOGRAMS
                (
                    train_sig_hist,
                    train_bkg_hist,
                    test_sig_hist,
                    test_bkg_hist,
                    bins,
                ) = model_response_hists(
                    y_train_sig_pred,
                    y_train_bkg_pred,
                    y_test_sig_pred,
                    y_test_bkg_pred,
                    w_train_sig,
                    w_train_bkg,
                    w_test_sig,
                    w_test_bkg,
                    normalize=False,
                )

                # print sum of all histograms to check normalization
                logging.info(f"Checking normalization of response histograms ...")
                log_histo_weights(
                    train_sig_hist,
                    train_bkg_hist,
                    test_sig_hist,
                    test_bkg_hist,
                    sig_label,
                )

                # SAVE RESULTS TO CSV
                save_results_to_csv(
                    bins,
                    test_sig_hist,
                    test_bkg_hist,
                    train_sig_hist,
                    train_bkg_hist,
                    method,
                    results_dir,
                )
                # FEATURE IMPORTANCE
                if method == "XGBoost":
                    importances = model.feature_importances_
                    var_names = good_vars
                    importance_dict = {"variables": good_vars}
                    importance_dict[f"importance_{sig_label}"] = importances
                    importance_df = pd.DataFrame(importance_dict)

                    # sort by importance
                    importance_df.sort_values(
                        by=f"importance_{sig_label}",
                        ascending=False,
                        inplace=True,
                        ignore_index=True,
                    )
                    importance_df.to_csv(f"{results_dir}/importance.csv", index=False)
                    print(f"Importance saved to {results_dir}/importance.csv")
