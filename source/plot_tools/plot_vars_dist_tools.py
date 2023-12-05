import numpy as np
import matplotlib.pyplot as plt
import awkward as ak


def plot_var_dist(
    signal,
    backgrounds,
    signal_weight,
    background_weights,
    sig_labels,
    bkg_labels,
    varname,
    sig_index,
    out_dir,
):
    if "tkIso_R03" in varname:
        plt.xlim(0, 60)
        bins = np.linspace(0, 60, 100)

    elif "nValid" in varname:
        xmax = max(signal[varname][sig_index])
        #round xmax up to largest integer
        xmax = int(np.ceil(xmax))
        plt.xlim(0, xmax)
        bins = np.linspace(0, xmax, xmax+1)

    elif "cos" in varname:
        plt.xlim(-1, 1)
        bins = np.linspace(-1, 1, 40)

    else:
        #use 1% and 99% percentile to set x limits
        xmin = np.percentile(signal[varname][sig_index], 1)
        xmax = np.percentile(signal[varname][sig_index], 99)
        if xmin == xmax:
            xmin -= 1
            xmax += 1
            bins = np.linspace(xmin, xmax, 3)
        bins = np.linspace(xmin, xmax, 100)
        plt.xlim(xmin, xmax)


    plt.hist(
        signal[varname][sig_index],
        bins,
        weights=signal_weight[varname][sig_index],
        alpha=1,
        density=True,
        label=sig_labels[sig_index],
        histtype="step",
        color="black",
    )
    plt.hist(
        backgrounds[varname],
        bins,
        weights=background_weights[varname],
        alpha=0.5,
        density=True,
        stacked=True,
        label=bkg_labels,
    )

    tot_sig_weight = np.nansum(signal_weight[varname][sig_index])
    tot_bkg_weight = 0
    for bkg_weight in background_weights[varname]:
        tot_bkg_weight += np.sum(bkg_weight)

    plt.xlabel(varname)
    plt.ylabel("Events")
    plt.legend()
    plt.savefig(f"{out_dir}/{varname}.png")
    plt.close()


def load_variables(vars, strees, btrees, weight_name):
    sig = {}
    bkg = {}
    sig_weight = {}
    bkg_weight = {}

    data_sig = [stree.arrays(vars) for stree in strees]
    data_bkg = [btree.arrays(vars) for btree in btrees]
    weight_sig = [stree.arrays(weight_name) for stree in strees]
    weight_bkg = [btree.arrays(weight_name) for btree in btrees]

    print("Loading Variables...")
    for h in vars:
        ##background
        ba = [data_bkg[i][h] for i in range(len(data_bkg))]
        bw = [weight_bkg[i][weight_name] for i in range(len(weight_bkg))]
        # broadcast bw with ba to match the shape
        bw = [ak.broadcast_arrays(bw[i], ba[i])[0] for i in range(len(bw))]

        bkg[h] = [ak.to_numpy(ak.flatten(ba[i])) for i in range(len(ba))]
        bkg_weight[h] = [ak.to_numpy(ak.flatten(bw[i])) for i in range(len(bw))]

        ##signal
        sa = [data_sig[i][h] for i in range(len(data_sig))]
        sw = [weight_sig[i][weight_name] for i in range(len(weight_sig))]
        # broadcast sw with sa to match the shape
        sw = [ak.broadcast_arrays(sw[i], sa[i])[0] for i in range(len(sw))]

        sig[h] = [ak.to_numpy(ak.flatten(sa[i])) for i in range(len(sa))]
        sig_weight[h] = [ak.to_numpy(ak.flatten(sw[i])) for i in range(len(sw))]
    print("Variables Loaded!")
    return sig, bkg, sig_weight, bkg_weight
