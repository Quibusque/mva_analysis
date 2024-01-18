from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
import pandas as pd


import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization
from sklearn.metrics import log_loss


from cfg.hnl_mva_tools import read_json_file
from sklearn.metrics import accuracy_score

# ┌─────────────────────────────────────┐
# │ THIS IS THE LIST OF ALLOWED METHODS │
# └─────────────────────────────────────┘
methods_list = ["XGBoost", "adaboost", "keras_shallow", "keras_deep"]


def build_and_train_xgboost(
    x_train,
    y_train,
    w_train,
    max_depth,
    n_estimators,
    early_stopping_rounds,
    output_path,
    eval_metric,
    objective,
    x_val=None,
    y_val=None,
    w_val=None,
):
    must_validate = all([x_val is not None, y_val is not None, w_val is not None])

    if must_validate:
        es = EarlyStopping(
            rounds=early_stopping_rounds,
            min_delta=5e-3,
            save_best=True,
            maximize=False,
            data_name="validation_0",
            metric_name=eval_metric,
        )
        bst = XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            objective=objective,
            callbacks=[es],
        )

        bst.fit(
            x_train,
            y_train,
            sample_weight=w_train,
            eval_set=[(x_val, y_val)],
            sample_weight_eval_set=[w_val],
            verbose=True,
        )
    else:
        bst = XGBClassifier(
            max_depth=max_depth,
            n_estimators=n_estimators,
            eval_metric=eval_metric,
            objective=objective,
        )

        bst.fit(x_train, y_train, sample_weight=w_train)

    # THIS WAS FOR DEBUGGING
    # print("bst: ", bst)
    # print("bst.best_iteration: ", bst.best_iteration)

    # # print the bst predict on signal validation dataset
    # # also print the first variable of x_val for the
    # # corresponding event
    # first_elements = x_val[:, 0]
    # # make it first 10 elements
    # first_elements = first_elements[:10]
    # predictions = bst.predict(x_val, output_margin=True)
    # predictions = predictions[:10]
    # for i in range(len(first_elements)):
    #     print(
    #         f"Event {i} has first variable {first_elements[i]} and prediction {predictions[i]}"
    #     )

    # print("SECOND TRAIN WITH DIFF STUFF")
    # bst_2 = XGBClassifier(
    #     max_depth=max_depth,
    #     n_estimators=n_estimators,
    #     eval_metric=eval_metric,
    #     objective=objective,
    #     early_stopping_rounds=early_stopping_rounds,
    # )
    # bst_2.fit(
    #     x_train,
    #     y_train,
    #     sample_weight=w_train,
    #     eval_set=[(x_val, y_val)],
    #     sample_weight_eval_set=[w_val],
    #     verbose=True,
    # )
    # predictions = bst_2.predict(x_val, output_margin=True)
    # predictions = predictions[:10]
    # for i in range(len(first_elements)):
    #     print(
    #         f"Event {i} has first variable {first_elements[i]} and prediction {predictions[i]}"
    #     )

    if output_path is not None:
        if not output_path.endswith(".txt"):
            output_path += ".txt"
        bst.save_model(output_path)
        print(f"Model saved to {output_path}")

    # param_grid = {
    #     "eta": [0.01, 0.05, 0.1, 0.2, 0.3],
    #     "max_depth": [3, 4, 5, 6],

    # }
    # scoring = {'AUC': 'roc_auc', 'F1': 'f1'}

    # optimal_params = GridSearchCV(
    #     estimator=bst,
    #     param_grid=param_grid,
    #     scoring=scoring,
    #     refit='F1',
    #     verbose=0,
    #     n_jobs=4,
    #     cv=3,
    # )

    # optimal_params.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    # # Print the best parameters
    # print("Best parameters: ", optimal_params.best_params_)

    # # Print the AUC and F1 for each parameter combination
    # results = pd.DataFrame(optimal_params.cv_results_)
    # print("Results: ##########")
    # print(results[['mean_test_AUC', 'mean_test_F1', 'params']])

    return bst


def build_and_train_adaboost_model(
    x_train,
    y_train,
    w_train,
    max_depth,
    min_samples_leaf,
    n_estimators,
    min_n_estimators,
    early_stopping_rounds,
    output_path,
    x_val=None,
    y_val=None,
    w_val=None,
    verbose=True,
):
    must_validate = all([x_val is not None, y_val is not None, w_val is not None])

    if must_validate:
        best_loss = np.inf
        best_n_estimators = 0
        no_improvement_counter = 0
        best_bdt = None
        for i in range(min_n_estimators, n_estimators + 1, 2):
            bdt = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(
                    max_depth=max_depth, min_samples_leaf=min_samples_leaf
                ),
                n_estimators=i,
            )
            bdt.fit(
                x_train,
                y_train,
                sample_weight=w_train,
            )
            y_pred_val = bdt.predict_proba(x_val)[:, 1]

            loss = log_loss(y_val, y_pred_val, sample_weight=w_val)
            if verbose:
                print(f"n_estimators={i}, loss={loss}")
            if loss < best_loss:
                best_loss = loss
                best_n_estimators = i
                no_improvement_counter = 0
                best_bdt = bdt
            else:
                no_improvement_counter += 1

            if no_improvement_counter >= early_stopping_rounds:
                bdt = best_bdt
                print(
                    f"Early stopping at n_estimators={best_n_estimators} with loss={best_loss}"
                )
                break

    else:
        bdt = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=max_depth),
            n_estimators=n_estimators,
        )
        bdt.fit(x_train, y_train, sample_weight=w_train)

    if output_path is not None:
        if not output_path.endswith(".joblib"):
            output_path += ".joblib"
        joblib.dump(bdt, output_path)
        print(f"Model saved to {output_path}")

    return bdt


def build_dense_keras_model(
    input_shape,
    hidden_layers,
    output_path,
):
    # Define the model
    model = Sequential()

    # add a normalization layer as first layer
    model.add(Normalization(input_shape=input_shape))

    # Add the hidden layers
    for layer_size in hidden_layers:
        model.add(Dense(layer_size, activation="relu"))

    # Add the output layer
    model.add(Dense(1, activation="sigmoid"))

    # Compile the model
    model.compile(
        loss="binary_crossentropy", optimizer="adam", weighted_metrics=["accuracy"]
    )
    if output_path is not None:
        # if output_file does not have a .h5 extension, add it
        if not output_path.endswith(".h5"):
            output_path += ".h5"
        model.save(output_path)
        print(f"Model saved to {output_path}")
    return model


def build_and_save_dense_keras_model_for_TMVA(
    input_shape,
    hidden_layers,
    output_path,
):
    # Define the model
    model = Sequential()

    # add a normalization layer as first layer
    model.add(Normalization(input_shape=input_shape))

    # Add the hidden layers
    for layer_size in hidden_layers:
        model.add(Dense(layer_size, activation="relu"))

    # Add the output layer, two nodes for signal and background
    model.add(Dense(2, activation="softmax"))

    # Compile the model
    model.compile(
        loss="binary_crossentropy", optimizer="adam", weighted_metrics=["accuracy"]
    )

    if output_path is not None:
        # if output_file does not have a .h5 extension, add it
        if not output_path.endswith(".h5"):
            output_path += ".h5"
        model.save(output_path)
        print(f"Model saved to {output_path}")
    return model


def train_compiled_dense_keras_model(
    x_train,
    y_train,
    w_train,
    model,
    epochs,
    batch_size,
    output_path,
    x_val=None,
    y_val=None,
    w_val=None,
    early_stopping_rounds=10,
):
    must_validate = all([x_val is not None, y_val is not None, w_val is not None])

    if must_validate:
        model.fit(
            x_train,
            y_train,
            sample_weight=w_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            validation_data=(x_val, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=early_stopping_rounds
                )
            ],
        )
    else:
        model.fit(
            x_train,
            y_train,
            sample_weight=w_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
        )

    if output_path is not None:
        # if output_file does not have a .h5 extension, add it
        if not output_path.endswith(".h5"):
            output_path += ".h5"
        model.save(output_path)
        print(f"Model saved to {output_path}")
    return model


def load_xgboost_model(model_path):
    # if model path does not have .txt extension, add it
    if not model_path.endswith(".txt"):
        model_path += ".txt"
    bst = XGBClassifier()
    bst.load_model(model_path)
    return bst


def load_adaboost_model(model_path):
    # if model path does not have .joblib extension, add it
    if not model_path.endswith(".joblib"):
        model_path += ".joblib"
    return joblib.load(model_path)


def load_dense_keras_model(model_path):
    # if model path does not have .keras extension or .h5 extension, add it
    if not model_path.endswith(".keras") and not model_path.endswith(".h5"):
        model_path += ".h5"
    return tf.keras.saving.load_model(model_path)


def train_one_signal_one_method(
    x_train,
    y_train,
    w_train,
    method,
    out_dir,
    xgboost_eval_metric,
    xgboost_objective,
    x_val=None,
    y_val=None,
    w_val=None,
    hyperpars_file: str = "source/cfg/hyperparameters.json",
):
    # if method not in methods_list, raise ValueError
    if method not in methods_list:
        raise ValueError("method not in methods_list")

    # x_val, y_val, w_val are optional. Either all or none of them must be given
    if (x_val is None) != (y_val is None) or (y_val is None) != (w_val is None):
        raise ValueError("Either all or none of x_val, y_val, w_val must be given")

    must_validate = all([x_val is not None, y_val is not None, w_val is not None])

    hyperpars = read_json_file(hyperpars_file)

    # TRAIN
    output_path = f"{out_dir}/{method}_model"

    # ┌─────────────────┐
    # │ WITH VALIDATION │
    # └─────────────────┘
    if must_validate:
        if method == "XGBoost":
            # HYPERPARAMETERS #
            max_depth = hyperpars["XGBoost"]["max_depth"]
            n_estimators = hyperpars["XGBoost"]["n_estimators"]
            early_stopping_rounds = hyperpars["XGBoost"]["early_stopping_rounds"]
            print(
                f"Training method: {method} with max_depth={max_depth} and n_estimators={n_estimators} ..."
            )
            model = build_and_train_xgboost(
                x_train,
                y_train,
                w_train,
                max_depth,
                n_estimators,
                early_stopping_rounds,
                output_path,
                xgboost_eval_metric,
                xgboost_objective,
                x_val,
                y_val,
                w_val,
            )
            print("Model trained!")
        elif method == "adaboost":
            # HYPERPARAMETERS #
            max_depth = hyperpars["adaboost"]["max_depth"]
            min_samples_leaf = hyperpars["adaboost"]["min_samples_leaf"]
            n_estimators = hyperpars["adaboost"]["n_estimators"]
            early_stopping_rounds = hyperpars["adaboost"]["early_stopping_rounds"]
            min_n_estimators = hyperpars["adaboost"]["min_n_estimators"]
            # min_samples_leaf = hyperpars["adaboost"]["min_samples_leaf"]
            print(
                f"Training method: {method} with max_depth={max_depth} and n_estimators={n_estimators} ..."
            )
            model = build_and_train_adaboost_model(
                x_train,
                y_train,
                w_train,
                max_depth,
                min_samples_leaf,
                n_estimators,
                min_n_estimators,
                early_stopping_rounds,
                output_path,
                x_val,
                y_val,
                w_val,
                verbose=True,
            )
            print("Model trained!")
        elif method == "keras_shallow":
            # HYPERPARAMETERS #
            epochs = hyperpars["keras_shallow"]["epochs"]
            batch_size = hyperpars["keras_shallow"]["batch_size"]
            hidden_layers = hyperpars["keras_shallow"]["hidden_layers"]
            early_stopping_rounds = hyperpars["keras_shallow"]["early_stopping_rounds"]

            input_shape = (x_train.shape[1],)
            print(
                f"Training method: {method} with {epochs} epochs and hidden_layer structure {hidden_layers} and {batch_size} batch size ..."
            )
            model = build_dense_keras_model(input_shape, hidden_layers, output_path)
            # if new_vars:
            #     tmva_output_path = hyperpars["keras_shallow"]["filename_model_new"]
            # else:
            #     tmva_output_path = hyperpars["keras_shallow"]["filename_model_old"]
            # build_and_save_dense_keras_model_for_TMVA(
            #     input_shape, hidden_layers, tmva_output_path
            # )
            train_compiled_dense_keras_model(
                x_train,
                y_train,
                w_train,
                model,
                epochs,
                batch_size,
                output_path,
                x_val,
                y_val,
                w_val,
                early_stopping_rounds,
            )

    # ┌────────────────────┐
    # │ WITHOUT VALIDATION │
    # └────────────────────┘
    else:
        if method == "XGBoost":
            # HYPERPARAMETERS #
            max_depth = hyperpars["XGBoost"]["max_depth"]
            n_estimators = hyperpars["XGBoost"]["n_estimators"]
            print(
                f"Training method: {method} with max_depth={max_depth} and n_estimators={n_estimators} ..."
            )
            output_path = f"{out_dir}/{method}_model"
            model = build_and_train_xgboost(
                x_train, y_train, w_train, max_depth, n_estimators, output_path
            )
            print("Model trained!")
        elif method == "adaboost":
            # HYPERPARAMETERS #
            max_depth = hyperpars["adaboost"]["max_depth"]
            n_estimators = hyperpars["adaboost"]["n_estimators"]
            min_n_estimators = hyperpars["adaboost"]["min_n_estimators"]
            print(
                f"Training method: {method} with max_depth={max_depth} and n_estimators={n_estimators} ..."
            )
            model = build_and_train_adaboost_model(
                x_train, y_train, w_train, max_depth, n_estimators, output_path
            )
            print("Model trained!")
        elif method == "keras_shallow":
            # HYPERPARAMETERS #
            epochs = hyperpars["keras_shallow"]["epochs"]
            batch_size = hyperpars["keras_shallow"]["batch_size"]
            hidden_layers = hyperpars["keras_shallow"]["hidden_layers"]

            input_shape = (x_train.shape[1],)
            print(
                f"Training method: {method} with {epochs} epochs and hidden_layer structure {hidden_layers} and {batch_size} batch size ..."
            )
            model = build_dense_keras_model(input_shape, hidden_layers, output_path)
            # if new_vars:
            #     tmva_output_path = hyperpars["keras_shallow"]["filename_model_new"]
            # else:
            #     tmva_output_path = hyperpars["keras_shallow"]["filename_model_old"]
            # build_and_save_dense_keras_model_for_TMVA(
            #     input_shape, hidden_layers, tmva_output_path
            # )

            train_compiled_dense_keras_model(
                x_train,
                y_train,
                w_train,
                model,
                epochs,
                batch_size,
                output_path,
            )

            print("Model trained!")
        elif method == "keras_deep":
            # HYPERPARAMETERS #
            epochs = hyperpars["keras_deep"]["epochs"]
            batch_size = hyperpars["keras_deep"]["batch_size"]
            hidden_layers = hyperpars["keras_deep"]["hidden_layers"]

            input_shape = (x_train.shape[1],)
            print(
                f"Training method: {method} with {epochs} epochs and hidden_layer structure {hidden_layers} and {batch_size} batch size ..."
            )
            model = build_dense_keras_model(input_shape, hidden_layers, output_path)
            print("Model trained!")


def train_one_signal_all_methods(
    x_train,
    y_train,
    w_train,
    methods_list,
    out_dir,
    xgboost_eval_metric="rmsle",
    xgboost_objective="binary:logistic",
    hyperpars_file: str = "source/cfg/hyperparameters.json",
    x_val=None,
    y_val=None,
    w_val=None,
):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for method in methods_list:
        train_one_signal_one_method(
            x_train,
            y_train,
            w_train,
            method,
            out_dir,
            xgboost_eval_metric,
            xgboost_objective,
            hyperpars_file=hyperpars_file,
            x_val=x_val,
            y_val=y_val,
            w_val=w_val,
        )


def load_model(model_path, method):
    # if method not in methods_list, raise ValueError
    if method not in methods_list:
        raise ValueError("method not in methods_list")

    if method == "XGBoost":
        return load_xgboost_model(model_path)
    elif method == "adaboost":
        return load_adaboost_model(model_path)
    elif method == "keras_shallow" or method == "keras_deep":
        return load_dense_keras_model(model_path)
