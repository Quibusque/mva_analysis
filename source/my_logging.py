import numpy as np
import logging


# LOGGING LEVEL
logging.basicConfig(level=logging.INFO)


def log_weights(y_train, y_test, w_train, w_test, sig_label, y_val=None, w_val=None):
    logging.info(
        f"Total weights for {sig_label} signal train: {np.sum(w_train[y_train == 1])}"
    )
    logging.info(
        f"Total weights for {sig_label} background train: {np.sum(w_train[y_train == 0])}"
    )
    logging.info(
        f"Total weights for {sig_label} signal test: {np.sum(w_test[y_test == 1])}"
    )
    logging.info(
        f"Total weights for {sig_label} background test: {np.sum(w_test[y_test == 0])}"
    )
    if (y_val is None) != (w_val is None):
        raise ValueError("Either all or none of y_val, w_val must be given")
    if y_val is not None and w_val is not None:
        logging.info(
            f"Total weights for {sig_label} signal validation: {np.sum(w_val[y_val == 1])}"
        )
        logging.info(
            f"Total weights for {sig_label} background validation: {np.sum(w_val[y_val == 0])}"
        )


def log_num_events(y_train, y_test, sig_label, y_val=None):
    logging.info(
        f"Total events for {sig_label} signal train: {len(y_train[y_train == 1])}"
    )
    logging.info(
        f"Total events for {sig_label} background train: {len(y_train[y_train == 0])}"
    )
    logging.info(
        f"Total events for {sig_label} signal test: {len(y_test[y_test == 1])}"
    )
    logging.info(
        f"Total events for {sig_label} background test: {len(y_test[y_test == 0])}"
    )
    if y_val is not None:
        logging.info(
            f"Total events for {sig_label} signal validation: {len(y_val[y_val == 1])}"
        )
        logging.info(
            f"Total events for {sig_label} background validation: {len(y_val[y_val == 0])}"
        )


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
            raise ValueError("Either all or none of val_sig_hist, val_bkg_hist must be given")
    if val_sig_hist is not None and val_bkg_hist is not None:
        logging.info(
            f"Total weights for {sig_label} signal validation: {np.sum(val_sig_hist)}"
        )
        logging.info(
            f"Total weights for {sig_label} background validation: {np.sum(val_bkg_hist)}"
        )