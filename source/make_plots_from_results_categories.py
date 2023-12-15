import uproot
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

from mva_tools.mva_response_tools import (
    tpr,
    fpr,
)


def plot_response_hists(
    train_sig_hist,
    train_bkg_hist,
    test_sig_hist,
    test_bkg_hist,
    bin_centers,
    method,
    sig_label,
    out_dir,
    log_scale: bool = False,
):
    # normalize the histograms
    train_sig_hist_norm = train_sig_hist / np.sum(train_sig_hist)
    train_bkg_hist_norm = train_bkg_hist / np.sum(train_bkg_hist)
    test_sig_hist_norm = test_sig_hist / np.sum(test_sig_hist)
    test_bkg_hist_norm = test_bkg_hist / np.sum(test_bkg_hist)

    bin_width = bin_centers[1] - bin_centers[0]

    # ax1 is the response histogram
    plt.step(
        bin_centers,
        train_sig_hist_norm,
        label="train sig",
        where="mid",
        color="tab:blue",
    )
    plt.step(
        bin_centers,
        train_bkg_hist_norm,
        label="train bkg",
        where="mid",
        color="tab:orange",
    )
    plt.bar(
        bin_centers,
        test_sig_hist_norm,
        label="test sig",
        alpha=0.5,
        width=bin_width,
        color="tab:blue",
    )
    plt.bar(
        bin_centers,
        test_bkg_hist_norm,
        label="test bkg",
        alpha=0.5,
        width=bin_width,
        color="tab:orange",
    )

    if log_scale:
        plt.yscale("log")

    plt.xlabel("Response")
    plt.ylabel("Normalized Counts")
    plt.title(f"{sig_label} {method} Response")

    plt.legend()

    # save plot
    plt.savefig(f"{out_dir}/{method}_response.png")
    plt.close()


if __name__ == "__main__":
    # parser args
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, help="input directory with results")
    parser.add_argument(
        "--out_dir_name",
        type=str,
        help="name of output directory for the plots",
        required=True,
    )

    args = parser.parse_args()

    input_dir = args.input_dir
    out_dir_name = args.out_dir_name

    # results/myMVA/siglabel0, results/myMVA/siglabel1 get siglabels
    sig_labels = os.listdir(f"{input_dir}/myMVA")

    methods_list = ["XGBoost"]
    categories = [1, 2, 3, 4, 5, 6]

    for sig_label in sig_labels:
        print(sig_label)
        myMVA_dir = f"{input_dir}/myMVA/{sig_label}"
        for category in categories:
            category_dir = f"{myMVA_dir}/cat_{category}"
            plots_dir = f"{category_dir}/{out_dir_name}"
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)

            # ┌──────┐
            # │ ROCS │
            # └──────┘

            for method in methods_list:
                print(f"method: {method}")
                df = pd.read_csv(f"{category_dir}/{method}_results.csv")

                # read {method}_results.csv
                tp_arr = np.array(df["tp"])
                fp_arr = np.array(df["fp"])
                fn_arr = np.array(df["fn"])
                tn_arr = np.array(df["tn"])

                # roc
                sig_eff = tpr(tp_arr, fn_arr)
                bkg_eff = np.array(fpr(fp_arr, tn_arr))
                x_values = sig_eff
                y_values = 1 - bkg_eff

                # it can happen that because of the specific values, sig_eff
                # does not actually go from 0 to 1, but stops to values > 0
                # this part is added so that the full curve is shown and is
                # especially important for the ROC integral

                min_x = np.min(x_values)
                if min_x > 1e-3:
                    new_x_values = np.linspace(min_x - 1e-3, 0, 10)
                    new_y_values = np.ones(10)

                    x_values = np.concatenate((x_values, new_x_values))
                    y_values = np.concatenate((y_values, new_y_values))

                # sort x and y in ascending order
                sort_idx = np.argsort(x_values)
                x_values = x_values[sort_idx]
                y_values = y_values[sort_idx]

                auc = np.trapz(y_values, x_values)

                plt.xlabel("Signal Efficiency")
                plt.ylabel("Background Rejection")

                # AXES LIMITS
                # X axis lower limit should be when y value is 0.95 of max y value
                x_lower_index = np.argmin(np.abs(y_values - 0.95))
                x_low = x_values[x_lower_index]
                x_high = 1.01
                plt.xlim(0.0, x_high)

                plt.plot(
                    x_values,
                    y_values,
                    label=f"{sig_label} {method} myMVA \nauc = {auc:.3f}",
                )

                plt.legend(loc="lower left")

                # save plot
                plt.savefig(f"{plots_dir}/{method}_roc.png")
                plt.close()

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
                    plots_dir,
                    log_scale=log_scale,
                )
