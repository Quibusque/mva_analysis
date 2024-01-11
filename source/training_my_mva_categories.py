import json, re, os
import numpy as np
import pandas as pd
import argparse
import logging

import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf

from cfg.hnl_mva_tools import read_json_file

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

from mva_tools.mva_plot_tools import plot_response_hists, compute_roc, plot_and_save_roc


methods_list = ["XGBoost"]
categories = [1, 2, 3, 4, 5, 6]
rng_seed = 10


if __name__ == "__main__":
    # parser args
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", action="store_true", help="train models")
    parser.add_argument("--results", action="store_true", help="produce results")
    parser.add_argument("--plots", action="store_true", help="make plots")
    # parser.add_argument("--validation", type=float, help="validation fraction")
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
        weight_name,
        sig_labels,
        bkg_labels,
    ) = read_files_and_open_trees(ntuples_json, vars_json)
    # LIST OF VARIABLES TO USE
    good_vars = read_json_file(vars_json)["training_vars"]
    scale_factor_vars = read_json_file(vars_json)["scale_factors"]
    if "C_pass_gen_matching" in good_vars:
        good_vars.remove("C_pass_gen_matching")

    print(f"file reading is done")
    # ┌────────────────────────────────┐
    # │ CHOOSE YOUR MASSES TO TRAIN ON │
    # └────────────────────────────────┘
    mass_list = [args.mass]
    my_sig_trees, my_sig_labels = filter_trees(
        sig_trees, sig_labels, mass_list=mass_list, ctau_list=["ctau10"]
    )

    # ┌───────────────────────────────┐
    # │ TEST AND VALIDATION FRACTIONS │
    # └───────────────────────────────┘
    test_fraction = 0.2
    validation_fraction = 0.1

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
            full_data = get_categorized_data(
                sig_tree,
                bkg_trees,
                good_vars,
                weight_name,
                test_fraction,
                rng_seed=rng_seed,
                equalnumevents=True,
                category_list=categories,
                validation_fraction=validation_fraction,
                scale_factor_vars=scale_factor_vars,
            )
            # TRAINING
            for category, data in zip(categories, full_data):
                category_dir = f"{results_dir}/cat_{category}"

                # make directory
                if not os.path.exists(category_dir):
                    os.makedirs(category_dir)

                print(f"Training category {category} ...")
                (
                    x_train,
                    x_val,
                    x_test,
                    y_train,
                    y_val,
                    y_test,
                    w_train,
                    w_val,
                    w_test,
                ) = data
                log_weights(y_train, y_test, w_train, w_test, sig_label, y_val, w_val)
                log_num_events(y_train, y_test, sig_label, y_val)
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

                # PLOT CORRELATION MATRIX
                plot_corr_matrix(
                    x_train_sig, x_test_sig, good_vars, category_dir, "sig"
                )
                plot_corr_matrix(
                    x_train_bkg, x_test_bkg, good_vars, category_dir, "bkg"
                )

                # TRAIN
                train_one_signal_all_methods(
                    x_train,
                    y_train,
                    w_train,
                    methods_list,
                    category_dir,
                    new_vars=True,
                    x_val=x_val,
                    y_val=y_val,
                    w_val=w_val,
                )
                print(f"Training category {category} complete!")
            print(f"Training {sig_label} complete!")
        print(f"Training complete!")
    # ┌─────────────────────────────────────────────────────────────────────────────────────┐
    # │ APPLY MODEL TO TEST SAMPLE AND GET VARIOUS RESULTS (IF ARGUMENT --results IS GIVEN) │
    # └─────────────────────────────────────────────────────────────────────────────────────┘
    if args.results:
        for sig_tree, sig_label in zip(my_sig_trees, my_sig_labels):
            print(f"Producing results for {sig_label} ...")
            results_dir = f"{out_dir}/{sig_label}"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            # GET DATA, CATEGORIZED
            full_data = get_categorized_data(
                sig_tree,
                bkg_trees,
                good_vars,
                weight_name,
                test_fraction,
                rng_seed=rng_seed,
                equalnumevents=True,
                category_list=categories,
                validation_fraction=validation_fraction,
                scale_factor_vars=scale_factor_vars,
            )
            # RESULTS
            for category, data in zip(categories, full_data):
                print(f"Producing results for category {category} ...")
                category_dir = f"{results_dir}/cat_{category}"

                (
                    x_train,
                    x_val,
                    x_test,
                    y_train,
                    y_val,
                    y_test,
                    w_train,
                    w_val,
                    w_test,
                ) = data
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

                for method in methods_list:
                    # PREDICT
                    model = load_model(f"{category_dir}/{method}_model", method)
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
                        normalize=True,
                        n_bins=40,
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
                        category_dir,
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
                        importance_df.to_csv(
                            f"{category_dir}/importance.csv", index=False
                        )
                        print(f"Importance saved to {category_dir}/importance.csv")
                print(f"Producing results for category {category} complete!")
            print(f"Producing results for {sig_label} complete!")
        print(f"Producing results complete!")
    # ┌────────────┐
    # │ MAKE PLOTS │
    # └────────────┘
    if args.plots:
        print(f"Making plots ...")
        for sig_label in my_sig_labels:
            print(f"Making plots for {sig_label} ...")
            results_dir = f"{out_dir}/{sig_label}"
            for category in categories:
                print(f"Making plots for category {category} ...")
                category_dir = f"{results_dir}/cat_{category}"
                plots_dir = f"{category_dir}/plots"
                if not os.path.exists(plots_dir):
                    os.makedirs(plots_dir)
                # ROC
                roc_data = []
                for method in methods_list:
                    df = pd.read_csv(f"{category_dir}/{method}_results.csv")

                    # ┌──────┐
                    # │ ROCS │
                    # └──────┘

                    # read {method}_results.csv
                    tp_arr = np.array(df["tp"])
                    fp_arr = np.array(df["fp"])
                    fn_arr = np.array(df["fn"])
                    tn_arr = np.array(df["tn"])

                    sig_eff, bkg_rej = compute_roc(tp_arr, fp_arr, fn_arr, tn_arr)
                    roc_data.append((sig_eff, bkg_rej))

                    # ┌─────────────────────┐
                    # │ RESPONSE HISTOGRAMS │
                    # └─────────────────────┘

                    log_scale = False
                    if "keras" in method:
                        log_scale = True

                    # response hists
                    bin_centers = np.array(df["bin_center"])
                    train_sig_hist = np.array(df["train_sig_hist"])
                    train_bkg_hist = np.array(df["train_bkg_hist"])
                    test_sig_hist = np.array(df["test_sig_hist"])
                    test_bkg_hist = np.array(df["test_bkg_hist"])

                    plot_response_hists(
                        train_sig_hist,
                        train_bkg_hist,
                        test_sig_hist,
                        test_bkg_hist,
                        bin_centers,
                        method,
                        sig_label,
                        category,
                        plots_dir,
                        log_scale=log_scale,
                    )
                plot_and_save_roc(
                    roc_data, methods_list, sig_label, category, plots_dir
                )
            print(f"Making plots for {sig_label} complete!")
        print(f"Making plots complete!")
