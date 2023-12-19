import xgboost
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

from .mva_response_tools import (
    tpr,
    fpr,
)
import json


def plot_corr_matrix(x_train, x_test, var_names, out_dir, sig_or_bkg: str):
    df = pd.DataFrame(np.concatenate((x_train, x_test)), columns=var_names)
    corr_matrix = df.corr() * 100
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(
        left=0.2, right=0.8, bottom=0.2, top=0.8
    )  # Adjust the subplot parameters
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".0f",
        cmap="coolwarm",
        xticklabels=var_names,
        yticklabels=var_names,
        cbar=False,
    )
    plt.savefig(f"{out_dir}/{sig_or_bkg}_corr_matrix.png", bbox_inches="tight")
    plt.close()


def plot_response_hists(
    train_sig_hist,
    train_bkg_hist,
    test_sig_hist,
    test_bkg_hist,
    bin_centers,
    method,
    sig_label,
    category,
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
    plt.title(f"{sig_label} {method} Response, category {category}")

    plt.legend()

    # save plot
    plt.savefig(f"{out_dir}/{method}_response.png")
    plt.close()


def compute_roc(tp_arr, fp_arr, fn_arr, tn_arr):
    # roc
    sig_eff = np.array(tpr(tp_arr, fn_arr))
    bkg_eff = np.array(fpr(fp_arr, tn_arr))
    x_values = sig_eff
    y_values = 1 - bkg_eff

    #if there is no point at (1,0) or (0,1), add it
    # (this is needed for the ROC curve auc to make sense)
    if (1.,0.) not in zip(x_values, y_values):
        x_values = np.append(x_values, 1)
        y_values = np.append(y_values, 0)
    if (0.,1.) not in zip(x_values, y_values):
        x_values = np.append(x_values, 0)
        y_values = np.append(y_values, 1)

    # Create a DataFrame
    df = pd.DataFrame({'x': x_values, 'y': y_values})
    # drop duplicates with same x, while keeping the highest y
    df = df.sort_values(['x', 'y'], ascending=[True, False]).drop_duplicates('x')

    #sort by ascending x
    df = df.sort_values(['x'], ascending=[True])

    new_x_values = df['x'].tolist()
    new_y_values = df['y'].tolist()

    return new_x_values, new_y_values


def plot_and_save_roc(roc_data, methods, sig_label, category, out_dir: str, xlim: tuple = (0.5, 1.02), cut_based_json: str = "source/cfg/cut_based.json"):

    with open(cut_based_json, "r") as file:
        cut_based_dict = json.load(file)
        cut_based_sig_eff = cut_based_dict[sig_label]["sig_eff"]
        cut_based_bkg_rej = 1 - cut_based_dict[sig_label]["bkg_eff"]
        plt.scatter(
            cut_based_sig_eff,
            cut_based_bkg_rej,
            label="cut-based",
            color="black",
            marker="x",
        )
        #also draw a thin grey horizontal line with y=cut_based_bkg_rej
        plt.axhline(y=cut_based_bkg_rej, color="black", linewidth=2, alpha=0.5)

    for method, (x_values, y_values) in zip(methods, roc_data):
        auc = np.trapz(y_values, x_values)
        plt.plot(
            x_values,
            y_values,
            label=f"{method} \nauc = {auc:.3f}",
            linewidth=2,
        )


        #make a scatter plot, but with small dots
        plt.scatter(
            x_values,
            y_values,
            s=10,
        )


    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Rejection")
    plt.xlim(xlim)
    plt.grid(True)
    plt.title(f"ROC Curves for {sig_label}, category {category}")
    plt.legend(loc="lower left")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()