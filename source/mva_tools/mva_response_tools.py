import tensorflow as tf
import sklearn
import xgboost
import numpy as np
import pandas as pd


def my_predict(model, x):
    """
    This function is used to predict the response of the model.
    The function is needed because different libraries have different
    syntax for predicting the response.

    Args:
        model: the model
        x: the data
    Returns:
        the response
    """
    if isinstance(model, xgboost.sklearn.XGBClassifier):
        return model.predict(x, output_margin=True)
    elif isinstance(model, sklearn.ensemble.AdaBoostClassifier):
        return model.predict_proba(x)[:, 1]
    elif isinstance(model, tf.keras.Sequential):
        return model.predict(x)[:, 0]
    else:
        print(f"unkown type in my_predict: {type(model)}")
        return None


def model_response_hists(
    y_train_sig_pred,
    y_train_bkg_pred,
    y_test_sig_pred,
    y_test_bkg_pred,
    w_train_sig,
    w_train_bkg,
    w_test_sig,
    w_test_bkg,
    n_bins: int = 150,
    normalize: bool = True,
):
    """
    This function calculates the response histograms for the training and
    testing data given the model predictions and weights.

    Args:
        y_train_sig_pred: the model predictions for the signal training data
        y_train_bkg_pred: the model predictions for the background training data
        y_test_sig_pred: the model predictions for the signal testing data
        y_test_bkg_pred: the model predictions for the background testing data
        w_train_sig: the weights for the signal training data
        w_train_bkg: the weights for the background training data
        w_test_sig: the weights for the signal testing data
        w_test_bkg: the weights for the background testing data
        n_bins: the number of bins for the histograms
        normalize: if True, the histograms are normalized to unit area
    """
    # xlimits
    x_low = min(
        min(y_train_sig_pred),
        min(y_train_bkg_pred),
        min(y_test_sig_pred),
        min(y_test_bkg_pred),
    )
    x_high = max(
        max(y_train_sig_pred),
        max(y_train_bkg_pred),
        max(y_test_sig_pred),
        max(y_test_bkg_pred),
    )

    # make numpy histograms
    train_sig_hist, bin_edges = np.histogram(
        y_train_sig_pred, bins=n_bins, range=(x_low, x_high), weights=w_train_sig
    )
    train_bkg_hist, _ = np.histogram(
        y_train_bkg_pred, bins=n_bins, range=(x_low, x_high), weights=w_train_bkg
    )
    test_sig_hist, _ = np.histogram(
        y_test_sig_pred, bins=n_bins, range=(x_low, x_high), weights=w_test_sig
    )
    test_bkg_hist, _ = np.histogram(
        y_test_bkg_pred, bins=n_bins, range=(x_low, x_high), weights=w_test_bkg
    )

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # normalize the histograms
    if normalize:
        train_sig_hist = train_sig_hist / np.sum(train_sig_hist)
        train_bkg_hist = train_bkg_hist / np.sum(train_bkg_hist)
        test_sig_hist = test_sig_hist / np.sum(test_sig_hist)
        test_bkg_hist = test_bkg_hist / np.sum(test_bkg_hist)

    return train_sig_hist, train_bkg_hist, test_sig_hist, test_bkg_hist, bin_centers


def calculate_classification_rates(test_sig_hist, test_bkg_hist):
    """
    This function, given a response histogram for signal and background,
    calculates the true positive (tp), false positive (fp), true negative (tn)
    and false negative (fn) rates for each bin.

    Args:
        test_sig_hist: the response histogram for the signal
        test_bkg_hist: the response histogram for the background
    Returns:
        tp_arr: the true positive rate for each bin
        fp_arr: the false positive rate for each bin
        tn_arr: the true negative rate for each bin
        fn_arr: the false negative rate for each bin
    """
    assert len(test_sig_hist) == len(test_bkg_hist)
    n_bins = len(test_sig_hist)

    tp_arr = np.zeros(n_bins)
    fp_arr = np.zeros(n_bins)
    tn_arr = np.zeros(n_bins)
    fn_arr = np.zeros(n_bins)

    for i in range(n_bins):
        tp_arr[i] = np.sum(test_sig_hist[i:])
        fp_arr[i] = np.sum(test_bkg_hist[i:])
        tn_arr[i] = np.sum(test_bkg_hist[:i])
        fn_arr[i] = np.sum(test_sig_hist[:i])

    return tp_arr, fp_arr, tn_arr, fn_arr


def calculate_significance(tp_arr, fp_arr):
    significance = np.divide(
        tp_arr,
        np.sqrt(tp_arr + fp_arr),
        where=((tp_arr + fp_arr) != 0),
        out=np.ones_like(tp_arr)*np.nan,
    )

    return significance


def tpr(tp_arr, fn_arr):
    return np.divide(
        tp_arr,
        tp_arr + fn_arr,
        where=((tp_arr + fn_arr) != 0),
        out=np.ones_like(tp_arr)*np.nan,
    )


def fpr(fp_arr, tn_arr):
    return np.divide(
        fp_arr,
        fp_arr + tn_arr,
        where=((fp_arr + tn_arr) != 0),
        out=np.ones_like(fp_arr)*np.nan,
    )


def precision(tp_arr, fp_arr):
    return np.divide(
        tp_arr,
        tp_arr + fp_arr,
        where=((tp_arr + fp_arr) != 0),
        out=np.ones_like(tp_arr)*np.nan,
    )


def save_results_to_csv(
    bin_centers,
    test_sig_hist,
    test_bkg_hist,
    train_sig_hist,
    train_bkg_hist,
    method,
    results_dir,
):
    results_df = pd.DataFrame(
        columns=[
            "bin_center",
            "tp",
            "fp",
            "tn",
            "fn",
            "test_sig_hist",
            "test_bkg_hist",
            "train_sig_hist",
            "train_bkg_hist",
            "sig",
            "acc",
        ]
    )
    # CLASSIFICATION RATES
    tp_arr, fp_arr, tn_arr, fn_arr = calculate_classification_rates(
        test_sig_hist, test_bkg_hist
    )

    # SIGNIFICANCE
    significance_arr = calculate_significance(tp_arr, fp_arr)

    # ACCURACY
    acc_arr = (tp_arr + tn_arr) / (tp_arr + tn_arr + fp_arr + fn_arr)

    # check that all arrays have the same length
    if not all(
        [
            len(tp_arr) == len(fp_arr),
            len(tp_arr) == len(tn_arr),
            len(tp_arr) == len(fn_arr),
            len(tp_arr) == len(test_sig_hist),
            len(tp_arr) == len(test_bkg_hist),
            len(tp_arr) == len(train_sig_hist),
            len(tp_arr) == len(train_bkg_hist),
            len(tp_arr) == len(significance_arr),
            len(tp_arr) == len(acc_arr),
        ]
    ):
        raise ValueError("Arrays have different lengths")

    # add data to results_df
    results_df["bin_center"] = bin_centers
    results_df["tp"] = tp_arr
    results_df["fp"] = fp_arr
    results_df["tn"] = tn_arr
    results_df["fn"] = fn_arr
    results_df["test_sig_hist"] = test_sig_hist
    results_df["test_bkg_hist"] = test_bkg_hist
    results_df["train_sig_hist"] = train_sig_hist
    results_df["train_bkg_hist"] = train_bkg_hist
    results_df["sig"] = significance_arr
    results_df["acc"] = acc_arr

    # ┌─────────────────────────────────────┐
    # │ SAVE RESULTS TO .CSV IN RESULTS_DIR │
    # └─────────────────────────────────────┘

    results_df.to_csv(f"{results_dir}/{method}_results.csv", index=False)
