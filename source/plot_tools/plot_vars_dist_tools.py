import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import awkward as ak
import uproot


def plot_var_dist(
    signal,
    backgrounds,
    signal_weight,
    background_weights,
    sig_label,
    bkg_labels,
    varname,
    out_dir,
):
    if "tkIso_R03" in varname:
        plt.xlim(0, 60)
        bins = np.linspace(0, 60, 100)

    elif "nValid" in varname:
        xmax = max(signal[varname])
        #round xmax up to largest integer
        xmax = int(np.ceil(xmax))
        plt.xlim(0, xmax)
        bins = np.linspace(0, xmax, xmax+1)

    elif "cos" in varname:
        plt.xlim(-1, 1)
        bins = np.linspace(-1, 1, 40)

    else:
        #use 1% and 99% percentile to set x limits
        xmin = np.percentile(signal[varname], 1)
        xmax = np.percentile(signal[varname], 99)
        if xmin == xmax:
            xmin -= 1
            xmax += 1
            bins = np.linspace(xmin, xmax, 3)
        bins = np.linspace(xmin, xmax, 50)
        plt.xlim(xmin, xmax)


    plt.hist(
        signal[varname],
        bins,
        weights=signal_weight[varname],
        alpha=1,
        density=True,
        label=sig_label,
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

    tot_sig_weight = np.nansum(signal_weight[varname])
    tot_bkg_weight = 0
    for bkg_weight in background_weights[varname]:
        tot_bkg_weight += np.sum(bkg_weight)

    plt.xlabel(varname)
    plt.ylabel("Normalized number of events")
    plt.legend()
    plt.savefig(f"{out_dir}/{varname}.png")
    plt.close()

def plot_2d_var_dist(
    signal,
    backgrounds,
    signal_weight,
    background_weights,
    varname1,
    varname2,
    out_dir,
):
    bins = [np.linspace(0, 1, 100), np.linspace(0, 1, 100)]
    for i,varname in enumerate([varname1,varname2]):
        if "tkIso_R03" in varname:
            bins[i] = np.linspace(0, 30, 100)
        elif "nValid" in varname:
            xmax = max(signal[varname])
            #round xmax up to largest integer
            xmax = int(np.ceil(xmax))
            bins[i] = np.linspace(0, xmax, xmax+1)
        elif "cos" in varname:
            bins[i] = np.linspace(-1, 1, 20)
        else:
            #use 1% and 99% percentile to set x limits
            xmin = np.percentile(signal[varname], 1)
            xmax = np.percentile(signal[varname], 99)
            if xmin == xmax:
                xmin -= 1
                xmax += 1
                bins[i] = np.linspace(xmin, xmax, 3)
            bins[i] = np.linspace(xmin, xmax, 20)

    #flatten the bkg arrays
    flat_bkg1 = np.array(ak.flatten(backgrounds[varname1]))
    flat_bkg2 = np.array(ak.flatten(backgrounds[varname2]))
    flat_bkg_weight1 = np.array(ak.flatten(background_weights[varname1]))


    # Plot 2D histogram for signal and create colorbar
    _, _, _, image_signal = plt.hist2d(
        signal[varname1],
        signal[varname2],
        bins=bins,
        weights=signal_weight[varname1],
        alpha=0.5,
        cmap='Reds'
    )
    plt.colorbar(image_signal, label='Signal')

    # Plot 2D histogram for backgrounds and create colorbar
    _, _, _, image_background = plt.hist2d(
        flat_bkg1,
        flat_bkg2,
        bins=bins,
        weights=flat_bkg_weight1,
        alpha=0.5,
        cmap='Blues'
    )
    plt.colorbar(image_background, label='Background')


    plt.xlabel(varname1)
    plt.ylabel(varname2)
    plt.savefig(f"{out_dir}/{varname1}_vs_{varname2}.png")
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


def load_bkg_data(bkg_trees,good_vars,weight_name, scale_factor_vars):
    bkg = {h:[] for h in good_vars}
    bkg_weight = {h:[] for h in good_vars}

    for bkg_tree in bkg_trees:
        data_bkg = bkg_tree.arrays(good_vars)
        weight_bkg = bkg_tree.arrays(weight_name)
        scale_factor = bkg_tree.arrays(scale_factor_vars)
        for h in good_vars:
            b_a = data_bkg[h]
            b_w = weight_bkg[weight_name]
            for sf in scale_factor_vars:
                b_w = b_w*scale_factor[sf]

            #broadcast b_w with b_a to match the shape
            b_w = ak.broadcast_arrays(b_w,b_a)[0]

            bkg[h].append(ak.to_numpy(ak.flatten(b_a)))
            bkg_weight[h].append(ak.to_numpy(ak.flatten(b_w)))

            
    print("Background Variables Loaded!")
    return bkg, bkg_weight

def load_sig_data(sig_tree,good_vars,scale_factor_vars):
    sig = {}
    sig_weight = {}
    print("Loading Signal Variables...")

    data_sig = sig_tree.arrays(good_vars)
    scale_factor = sig_tree.arrays(scale_factor_vars)
    weight_sig = np.ones_like(scale_factor[scale_factor_vars[0]])
    for sf in scale_factor_vars:
        weight_sig = weight_sig*scale_factor[sf]
    
    for h in good_vars:
        sig[h] = ak.to_numpy(ak.flatten(data_sig[h]))
        sig_weight[h] = ak.to_numpy(ak.flatten(weight_sig))
    print("Signal Variables Loaded!")
    return sig, sig_weight