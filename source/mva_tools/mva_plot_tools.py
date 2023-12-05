import xgboost
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_corr_matrix(x_train,x_test,var_names,out_dir,sig_or_bkg:str):

    df = pd.DataFrame(np.concatenate((x_train, x_test)), columns=var_names)
    corr_matrix = df.corr() * 100
    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(left=0.2, right=0.8, bottom=0.2, top=0.8)  # Adjust the subplot parameters
    sns.heatmap(corr_matrix, annot=True, fmt=".0f", cmap="coolwarm", xticklabels=var_names, yticklabels=var_names, cbar=False)
    plt.savefig(f"{out_dir}/{sig_or_bkg}_corr_matrix.png", bbox_inches="tight")
    plt.close()