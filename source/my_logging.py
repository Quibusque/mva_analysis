import numpy as np
import pandas as pd
import logging


# LOGGING LEVEL
logging.basicConfig(level=logging.INFO)


def log_weights_and_events(
    y_train, y_test, w_train, w_test, out_dir, y_val=None, w_val=None
):
    if (y_val is None) != (w_val is None):
        raise ValueError("Either all or none of y_val, w_val must be given")

    signal_train_weights = np.sum(w_train[y_train == 1])
    background_train_weights = np.sum(w_train[y_train == 0])
    signal_test_weights = np.sum(w_test[y_test == 1])
    background_test_weights = np.sum(w_test[y_test == 0])
    signal_val_weights = np.sum(w_val[y_val == 1]) if y_val is not None else None
    background_val_weights = np.sum(w_val[y_val == 0]) if y_val is not None else None
    signal_train_events = len(y_train[y_train == 1])
    background_train_events = len(y_train[y_train == 0])
    signal_test_events = len(y_test[y_test == 1])
    background_test_events = len(y_test[y_test == 0])
    signal_val_events = len(y_val[y_val == 1]) if y_val is not None else None
    background_val_events = len(y_val[y_val == 0]) if y_val is not None else None

    total_signal_events = (
        signal_train_events + signal_test_events + signal_val_events
        if y_val is not None
        else signal_train_events + signal_test_events
    )
    total_background_events = (
        background_train_events + background_test_events + background_val_events
        if y_val is not None
        else background_train_events + background_test_events
    )
    total_signal_weights = (
        signal_train_weights + signal_test_weights + signal_val_weights
        if y_val is not None
        else signal_train_weights + signal_test_weights
    )
    total_background_weights = (
        background_train_weights + background_test_weights + background_val_weights
        if y_val is not None
        else background_train_weights + background_test_weights
    )

    data = {
        "signal_events": [
            signal_train_events,
            signal_test_events,
            signal_val_events,
            total_signal_events,
        ],
        "signal_weights": [
            signal_train_weights,
            signal_test_weights,
            signal_val_weights,
            total_signal_weights,
        ],
        "background_events": [
            background_train_events,
            background_test_events,
            background_val_events,
            total_background_events,
        ],
        "background_weights": [
            background_train_weights,
            background_test_weights,
            background_val_weights,
            total_background_weights,
        ],
    }

    df = pd.DataFrame(data, index=["training", "test", "validation", "total"])

    logging.info(df)
    df.to_csv(f"{out_dir}/weights_and_events.csv")


def log_histo_weights(
    train_sig_hist,
    train_bkg_hist,
    test_sig_hist,
    test_bkg_hist,
    sig_label,
    val_sig_hist=None,
    val_bkg_hist=None,
):
    logging.info(
        f"Total weights for {sig_label} signal train: {np.sum(train_sig_hist)}"
    )
    logging.info(
        f"Total weights for {sig_label} background train: {np.sum(train_bkg_hist)}"
    )
    logging.info(f"Total weights for {sig_label} signal test: {np.sum(test_sig_hist)}")
    logging.info(
        f"Total weights for {sig_label} background test: {np.sum(test_bkg_hist)}"
    )
    if (val_sig_hist is None) != (val_bkg_hist is None):
        raise ValueError(
            "Either all or none of val_sig_hist, val_bkg_hist must be given"
        )
    if val_sig_hist is not None and val_bkg_hist is not None:
        logging.info(
            f"Total weights for {sig_label} signal validation: {np.sum(val_sig_hist)}"
        )
        logging.info(
            f"Total weights for {sig_label} background validation: {np.sum(val_bkg_hist)}"
        )
