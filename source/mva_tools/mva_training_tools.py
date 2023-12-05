from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib

import os

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization


from cfg.hnl_mva_tools import read_json_file

# ┌─────────────────────────────────────┐
# │ THIS IS THE LIST OF ALLOWED METHODS │
# └─────────────────────────────────────┘
methods_list = ["XGBoost", "adaboost", "keras_shallow", "keras_deep"]


def build_and_train_xgboost(
    x_train, y_train, w_train, max_depth, n_estimators, output_path
):
    # Fit xgboost model
    bst = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators)
    bst.fit(x_train, y_train, sample_weight=w_train)

    if output_path is not None:
        if not output_path.endswith(".txt"):
            output_path += ".txt"
        bst.save_model(output_path)
        print(f"Model saved to {output_path}")

    return bst


def build_and_train_adaboost_model(
    x_train, y_train, w_train, max_depth, n_estimators, output_path
):
    # Fit xgboost model
    bdt = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=max_depth), n_estimators=n_estimators
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
):
    model.fit(
        x_train,
        y_train,
        sample_weight=w_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
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
    new_vars: bool,
    hyperpars_file: str = "source/cfg/hyperparameters.json",
):
    # if method not in methods_list, raise ValueError
    if method not in methods_list:
        raise ValueError("method not in methods_list")

    hyperpars = read_json_file(hyperpars_file)

    # TRAIN
    output_path = f"{out_dir}/{method}_model"
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
        if new_vars:
            tmva_output_path = hyperpars["keras_shallow"]["filename_model_new"]
        else:
            tmva_output_path = hyperpars["keras_shallow"]["filename_model_old"]
        build_and_save_dense_keras_model_for_TMVA(
            input_shape, hidden_layers, tmva_output_path
        )

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
    new_vars: bool,
    hyperpars_file: str = "source/cfg/hyperparameters.json",
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
            new_vars=new_vars,
            hyperpars_file=hyperpars_file,
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
