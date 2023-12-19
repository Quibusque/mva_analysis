import uproot
import numpy as np
import pandas as pd
import argparse
import os
import os
import matplotlib.pyplot as plt

from mva_tools.mva_plot_tools import plot_response_hists, compute_roc, plot_and_save_roc


# def plot_response_hists(
#     train_sig_hist,
#     train_bkg_hist,
#     test_sig_hist,
#     test_bkg_hist,
#     bin_centers,
#     method,
#     sig_label,
#     out_dir,
#     log_scale: bool = False,
# ):
#     # normalize the histograms
#     train_sig_hist_norm = train_sig_hist / np.sum(train_sig_hist)
#     train_bkg_hist_norm = train_bkg_hist / np.sum(train_bkg_hist)
#     test_sig_hist_norm = test_sig_hist / np.sum(test_sig_hist)
#     test_bkg_hist_norm = test_bkg_hist / np.sum(test_bkg_hist)

#     bin_width = bin_centers[1] - bin_centers[0]

#     # ax1 is the response histogram
#     plt.step(
#         bin_centers,
#         train_sig_hist_norm,
#         label="train sig",
#         where="mid",
#         color="tab:blue",
#     )
#     plt.step(
#         bin_centers,
#         train_bkg_hist_norm,
#         label="train bkg",
#         where="mid",
#         color="tab:orange",
#     )
#     plt.bar(
#         bin_centers,
#         test_sig_hist_norm,
#         label="test sig",
#         alpha=0.5,
#         width=bin_width,
#         color="tab:blue",
#     )
#     plt.bar(
#         bin_centers,
#         test_bkg_hist_norm,
#         label="test bkg",
#         alpha=0.5,
#         width=bin_width,
#         color="tab:orange",
#     )

#     if log_scale:
#         plt.yscale("log")

#     plt.xlabel("Response")
#     plt.ylabel("Normalized Counts")
#     plt.title(f"{sig_label} {method} Response")

#     plt.legend()

#     # save plot
#     plt.savefig(f"{out_dir}/{method}_response.png")
#     plt.close()


# def compute_roc(tp_arr, fp_arr, fn_arr, tn_arr):
#     # roc
#     sig_eff = tpr(tp_arr, fn_arr)
#     bkg_eff = np.array(fpr(fp_arr, tn_arr))
#     x_values = sig_eff
#     y_values = 1 - bkg_eff

#     # sometimes there are several points with the same x value, so we need to remove duplicates
#     # keep the one with highest y value
#     new_x_values = []
#     new_y_values = []
#     for i, (x, y) in enumerate(zip(x_values, y_values)):
#         if x not in new_x_values:
#             new_x_values.append(x)
#             new_y_values.append(y)
#         else:
#             index = new_x_values.index(x)
#             if y > new_y_values[index]:
#                 new_y_values[index] = y

#     # sort the points in ascending order of x
#     new_x_values = new_x_values[::-1]
#     new_y_values = new_y_values[::-1]

#     return new_x_values, new_y_values


# def plot_and_save_roc(roc_data, methods, out_dir: str, xlim: tuple = (0.55, 1.02)):
#     for method, (x_values, y_values) in zip(methods, roc_data):
#         auc = np.trapz(y_values, x_values)
#         plt.plot(
#             x_values,
#             y_values,
#             label=f"{method} \nauc = {auc:.3f}",
#         )
#         plt.scatter(
#             x_values,
#             y_values,
#         )

#     plt.xlabel("Signal Efficiency")
#     plt.ylabel("Background Rejection")
#     plt.xlim(xlim)
#     plt.legend(loc="lower left")
#     plt.grid(True)

#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)

#     plt.savefig(os.path.join(out_dir, "roc_curve.png"))
#     plt.close()


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

    methods_list = ["adaboost","XGBoost"]
    categories = [1, 2, 3, 4, 5, 6]

    for sig_label in sig_labels:
        print(f"Making plots for {sig_label}...")
        myMVA_dir = f"{input_dir}/myMVA/{sig_label}"
        for category in categories:
            category_dir = f"{myMVA_dir}/cat_{category}"
            plots_dir = f"{category_dir}/{out_dir_name}"
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            #roc_data will be a list of tuples (sig_eff, bkg_rej)
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
            plot_and_save_roc(roc_data, methods_list,sig_label,category, plots_dir, xlim=(0, 1.02))
