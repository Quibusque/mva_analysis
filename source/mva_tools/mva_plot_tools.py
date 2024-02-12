import xgboost
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import seaborn as sns
import os
import mplhep as hep

plt.style.use(hep.style.ROOT)

from .mva_response_tools import (
    tpr,
    fpr,
    calculate_classification_rates,
    calculate_significance,
)
import json

sig_label_dict = {
    "mN1p0_ctau10": "$m_{N} = 1 GeV, c\\tau = 10 mm$",
    "mN1p0_ctau100": "$m_{N} = 1 GeV, c\\tau = 100 mm$",
    "mN1p0_ctau1000": "$m_{N} = 1 GeV, c\\tau = 1000 mm$",
    "mN1p25_ctau10": "$m_{N} = 1.25 GeV, c\\tau = 10 mm$",
    "mN1p25_ctau100": "$m_{N} = 1.25 GeV, c\\tau = 100 mm$",
    "mN1p25_ctau1000": "$m_{N} = 1.25 GeV, c\\tau = 1000 mm$",
    "mN1p5_ctau10": "$m_{N} = 1.5 GeV, c\\tau = 10 mm$",
    "mN1p5_ctau100": "$m_{N} = 1.5 GeV, c\\tau = 100 mm$",
    "mN1p5_ctau1000": "$m_{N} = 1.5 GeV, c\\tau = 1000 mm$",
    "mN1p8_ctau10": "$m_{N} = 1.8 GeV, c\\tau = 10 mm$",
    "mN1p8_ctau100": "$m_{N} = 1.8 GeV, c\\tau = 100 mm$",
    "mN1p8_ctau1000": "$m_{N} = 1.8 GeV, c\\tau = 1000 mm$",
}

category_dict = {
    1: "lowDisp_SS",
    2: "mediumDisp_SS",
    3: "highDisp_SS",
    4: "lowDisp_OS",
    5: "mediumDisp_OS",
    6: "highDisp_OS"
}

method_dict = {
    "XGBoost" : "XGB",
    "keras_shallow": "NN",
    "adaboost": "BDT"
}

def plot_corr_matrix(x_train, x_test, var_names, out_dir, sig_or_bkg: str):
    df = pd.DataFrame(np.concatenate((x_train, x_test)), columns=var_names)
    corr_matrix = df.corr() * 100
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
    significance=None,
):
    # normalize the histograms
    train_sig_hist_norm = train_sig_hist / np.sum(train_sig_hist)
    train_bkg_hist_norm = train_bkg_hist / np.sum(train_bkg_hist)
    test_sig_hist_norm = test_sig_hist / np.sum(test_sig_hist)
    test_bkg_hist_norm = test_bkg_hist / np.sum(test_bkg_hist)

    bin_width = bin_centers[1] - bin_centers[0]

    # make ax1 and ax2, ax2 is for significance
    # it's all just one plot, but with two y-axes

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # ax1 is the response histogram
    ax1.step(
        bin_centers,
        train_sig_hist_norm,
        label="train sig",
        where="mid",
        color="tab:blue",
    )
    ax1.step(
        bin_centers,
        train_bkg_hist_norm,
        label="train bkg",
        where="mid",
        color="tab:orange",
    )
    ax1.bar(
        bin_centers,
        test_sig_hist_norm,
        label="test sig",
        alpha=0.5,
        width=bin_width,
        color="tab:blue",
    )
    ax1.bar(
        bin_centers,
        test_bkg_hist_norm,
        label="test bkg",
        alpha=0.5,
        width=bin_width,
        color="tab:orange",
    )

    if log_scale:
        ax1.set_yscale("log")

    ax1.set_xlabel("Response")
    ax1.set_ylabel("Normalized Counts")

    # fig set title
    fig.suptitle(f"{sig_label_dict[sig_label]} {category_dict[category]} {method} score")

    wp = None
    if significance is not None:
        # find max significance
        max_sig = np.nanmax(significance)
        # find the response value at which max_sig occurs
        max_sig_response = bin_centers[np.nanargmax(significance)]
        max_sig_index = np.nanargmax(significance)
        # draw a vertical line at that response value
        ax1.axvline(
            x=max_sig_response,
            color="tab:green",
            linewidth=2,
            alpha=0.5,
            linestyle="dashed",
            label="best cut",
        )

        ax2.plot(
            bin_centers,
            significance,
            label=f"$S/\sqrt{{S+B}}$",
            color="tab:green",
        )
        ax2.tick_params(axis='y', labelcolor='green')
        # sig_eff is integral of test_sig_hist from max_sig_index to end
        # over integral of test_sig_hist
        sig_eff = np.sum(test_sig_hist[max_sig_index:]) / np.sum(test_sig_hist)
        # bkg_eff is integral of test_bkg_hist from max_sig_index to end
        # over integral of test_bkg_hist
        bkg_eff = np.sum(test_bkg_hist[max_sig_index:]) / np.sum(test_bkg_hist)
        bkg_rej = 1 - bkg_eff
        wp = (sig_eff, bkg_rej)

    # make a single legend for both axes
    handles, labels = ax1.get_legend_handles_labels()
    if significance is not None:
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles += handles2
        labels += labels2
    ax1.legend(handles, labels, loc="best")

    plt.savefig(f"{out_dir}/{method}_response.png", bbox_inches="tight")
    plt.close()

    return wp, max_sig_response


def compute_roc(tp_arr, fp_arr, fn_arr, tn_arr):
    # roc
    sig_eff = np.array(tpr(tp_arr, fn_arr))
    bkg_eff = np.array(fpr(fp_arr, tn_arr))
    x_values = sig_eff
    y_values = 1 - bkg_eff

    # round to 4 decimal places
    x_values = np.round(x_values, 4)
    y_values = np.round(y_values, 4)

    # if there is no point at (1,0) or (0,1), add it
    # (this is needed for the ROC curve auc to make sense)
    if (1.0, 0.0) not in zip(x_values, y_values):
        x_values = np.append(x_values, 1)
        y_values = np.append(y_values, 0)
    if (0.0, 1.0) not in zip(x_values, y_values):
        x_values = np.append(x_values, 0)
        y_values = np.append(y_values, 1)

    # Create a DataFrame
    df = pd.DataFrame({"x": x_values, "y": y_values})
    # drop duplicates with same x, while keeping the highest y
    df = df.sort_values(["x", "y"], ascending=[True, False]).drop_duplicates("x")

    # sort by ascending x
    df = df.sort_values(["x"], ascending=[True])

    new_x_values = df["x"].tolist()
    new_y_values = df["y"].tolist()

    return new_x_values, new_y_values


def plot_and_save_roc(
    roc_data,
    methods,
    sig_label,
    category,
    out_dir: str,
    xlim: tuple = (0.5, 1.02),
    working_point=None,
    cut_based_json: str = "source/cfg/cut_based.json",
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, method, (x_values, y_values) in zip(colors, methods, roc_data):
        auc = np.trapz(y_values, x_values)
        plt.plot(
            x_values,
            y_values,
            label=f"{method} \nauc = {auc:.3f}",
            linewidth=2,
            color=color,
        )
        if method == "XGBoost" and working_point is not None:
            plt.scatter(
                working_point[0],
                working_point[1],
                label="working point",
                marker="o",
                s=30,
                color=color,
            )

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
            s=100,
        )
        # # also draw a thin grey horizontal line with y=cut_based_bkg_rej
        # plt.axhline(y=cut_based_bkg_rej, color="black", linewidth=2, alpha=0.5)
        # # also draw a thin grey vertical line with x=cut_based_sig_eff
        # plt.axvline(x=cut_based_sig_eff, color="black", linewidth=2, alpha=0.5)

    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Rejection")
    plt.xlim(xlim)
    plt.grid(True)
    plt.title(f"ROC Curves for {sig_label_dict[sig_label]} {category_dict[category]}")
    plt.legend(loc="lower left")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, "roc_curve.png"))
    plt.close()


def plot_loss_history_xgb(evals_dict, eval_metric, out_dir):
    eval_metric_name_dict = {"rmse": "RMSE"}
    val_loss = np.array(evals_dict[f"val_{eval_metric}"])
    train_loss = np.array(evals_dict[f"train_{eval_metric}"])
    # fin best iter as minimum loss
    best_iter = np.argmin(val_loss)

    n_trees = len(val_loss)
    plt.plot(
        np.arange(1, n_trees + 1),
        val_loss,
        label=f"Validation {eval_metric_name_dict[eval_metric]}",
        lw=2,
    )
    plt.plot(
        np.arange(1, n_trees + 1),
        train_loss,
        label=f"Training {eval_metric_name_dict[eval_metric]}",
        lw=2,
    )
    # plot dot at best iteration instead of line
    # plt.scatter(best_iter + 1, val_loss[best_iter], color="black", label=f"Best Iteration")
    plt.axvline(x=best_iter+1, color="black", linestyle="--", label=f"Best Iteration")
    plt.xlabel("Number of Trees")
    plt.ylabel(f"{eval_metric_name_dict[eval_metric]}")
    plt.title(f"XGB {eval_metric_name_dict[eval_metric]} History")
    plt.legend(loc="best")
    plt.grid(True)

    ax = plt.gca()  # get current axes
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # ensure x-axis has integer values

    plt.savefig(f"{out_dir}/xgb_{eval_metric}_history.png")
    plt.close()


def plot_loss_history_adaboost(evals_dict, out_dir):
    eval_metric_name_dict = {"logloss": "Log Loss"}
    eval_metric = "logloss"
    val_loss_history = np.array(evals_dict[f"val_{eval_metric}"])
    train_loss_history = np.array(evals_dict[f"train_{eval_metric}"])
    n_trees_history = np.array(evals_dict["n_trees"])
    best_iter = np.argmin(val_loss_history)
    best_n_trees = n_trees_history[best_iter]
    print(f"Best iteration: {best_iter}")
    print(f"history: {n_trees_history}")
    print(f"Best n_trees: {best_n_trees}")

    plt.plot(
        n_trees_history,
        val_loss_history,
        label=f"Validation {eval_metric_name_dict[eval_metric]}",
        lw=2,
    )
    plt.plot(
        n_trees_history,
        train_loss_history,
        label=f"Training {eval_metric_name_dict[eval_metric]}",
        lw=2,
    )
    plt.axvline(x=best_n_trees, color="black", linestyle="--", label=f"Best Iteration")
    # # plot dot at best iteration instead of line
    plt.xlabel("Number of trees")
    plt.ylabel(f"{eval_metric_name_dict[eval_metric]}")
    plt.title(f"BDT {eval_metric_name_dict[eval_metric]} History")
    plt.legend(loc="best")
    plt.grid(True)

    ax = plt.gca()  # get current axes
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # ensure x-axis has integer values

    plt.savefig(f"{out_dir}/adaboost_{eval_metric}_history.png")
    plt.close()


#     @@@@@ HISTORY @@@@@
# {'loss': [4.489935874938965, 0.6985096335411072, 0.5140586495399475,
# 0.44289177656173706, 0.44318386912345886, 0.435077965259552,
# 0.4280102252960205, 0.4546031951904297, 0.4028034508228302], 'val_loss':
# [0.719211995601654, 0.5138530731201172, 0.4081213176250458,
# 0.4484078884124756, 0.3862873911857605, 0.3464757800102234,
# 0.34257081151008606, 0.3854263126850128, 0.3595277667045593]}


def plot_loss_history_keras(history, out_dir):
    eval_metric = "loss"
    loss = history[eval_metric]
    val_loss = history[f"val_{eval_metric}"]
    best_epoch = np.argmin(val_loss)
    epochs = range(0, len(loss))
    eval_metric_name_dict = {"loss": "Log Loss"}
    # plt.plot(epochs, loss, label=f"Training {eval_metric_name_dict[eval_metric]}", lw=2)
    plt.plot(
        epochs, val_loss, label=f"Validation {eval_metric_name_dict[eval_metric]}", lw=2
    )
    plt.axvline(x=best_epoch, color="black", linestyle="--", label=f"Best Iteration")
    # plot dot at best iteration instead of line
    # plt.scatter(
    #     best_epoch, val_loss[best_epoch], color="black", label=f"Best Iteration"
    # )
    plt.title(f"NN {eval_metric_name_dict[eval_metric]} History")
    plt.xlabel("Epochs")
    plt.ylabel(f"{eval_metric_name_dict[eval_metric]}")
    plt.legend()
    plt.grid(True)

    ax = plt.gca()  # get current axes
    ax.xaxis.set_major_locator(
        MaxNLocator(integer=True)
    )  # ensure x-axis has integer values

    plt.savefig(f"{out_dir}/keras_{eval_metric}_history.png")
    plt.close()
