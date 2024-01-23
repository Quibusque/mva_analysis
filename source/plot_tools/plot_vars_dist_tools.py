import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import awkward as ak
import uproot

from data_tools.load_data import categorize_data


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
        xmax = max(signal)
        #round xmax up to largest integer
        xmax = int(np.ceil(xmax))
        plt.xlim(0, xmax)
        bins = np.linspace(0, xmax, xmax+1)

    elif "cos" in varname:
        plt.xlim(-1, 1)
        bins = np.linspace(-1, 1, 40)
    elif "charge" in varname:
        plt.xlim(-2, 2)
        bins = np.linspace(-2, 2, 5)

    else:
        #use 1% and 99% percentile to set x limits
        xmin = np.percentile(signal, 1)
        xmax = np.percentile(signal, 99)
        if xmin == xmax:
            xmin -= 1
            xmax += 1
            bins = np.linspace(xmin, xmax, 3)
        bins = np.linspace(xmin, xmax, 50)
        plt.xlim(xmin, xmax)


    plt.hist(
        signal,
        bins,
        weights=signal_weight,
        alpha=1,
        density=True,
        label=sig_label,
        histtype="step",
        color="black",
    )
    plt.hist(
        backgrounds,
        bins,
        weights=background_weights,
        alpha=0.5,
        density=True,
        stacked=True,
        label=bkg_labels,
    )

    tot_sig_weight = np.nansum(signal_weight)
    tot_bkg_weight = 0
    for bkg_weight in background_weights:
        tot_bkg_weight += np.sum(bkg_weight)

    plt.xlabel(varname)
    plt.ylabel("Normalized number of events")
    plt.legend()
    plt.savefig(f"{out_dir}/{varname}.png")
    plt.close()

# def plot_2d_var_dist(
#     signal,
#     backgrounds,
#     signal_weight,
#     background_weights,
#     varname1,
#     varname2,
#     out_dir,
# ):
#     bins = [np.linspace(0, 1, 100), np.linspace(0, 1, 100)]
#     for i,varname in enumerate([varname1,varname2]):
#         if "tkIso_R03" in varname:
#             bins[i] = np.linspace(0, 30, 100)
#         elif "nValid" in varname:
#             xmax = max(signal[varname])
#             #round xmax up to largest integer
#             xmax = int(np.ceil(xmax))
#             bins[i] = np.linspace(0, xmax, xmax+1)
#         elif "cos" in varname:
#             bins[i] = np.linspace(-1, 1, 20)
#         else:
#             #use 1% and 99% percentile to set x limits
#             xmin = np.percentile(signal[varname], 1)
#             xmax = np.percentile(signal[varname], 99)
#             if xmin == xmax:
#                 xmin -= 1
#                 xmax += 1
#                 bins[i] = np.linspace(xmin, xmax, 3)
#             bins[i] = np.linspace(xmin, xmax, 20)

#     #flatten the bkg arrays
#     flat_bkg1 = np.array(ak.flatten(backgrounds[varname1]))
#     flat_bkg2 = np.array(ak.flatten(backgrounds[varname2]))
#     flat_bkg_weight1 = np.array(ak.flatten(background_weights[varname1]))


#     # Plot 2D histogram for signal and create colorbar
#     _, _, _, image_signal = plt.hist2d(
#         signal[varname1],
#         signal[varname2],
#         bins=bins,
#         weights=signal_weight[varname1],
#         alpha=0.5,
#         cmap='Reds'
#     )
#     plt.colorbar(image_signal, label='Signal')

#     # Plot 2D histogram for backgrounds and create colorbar
#     _, _, _, image_background = plt.hist2d(
#         flat_bkg1,
#         flat_bkg2,
#         bins=bins,
#         weights=flat_bkg_weight1,
#         alpha=0.5,
#         cmap='Blues'
#     )
#     plt.colorbar(image_background, label='Background')


#     plt.xlabel(varname1)
#     plt.ylabel(varname2)
#     plt.savefig(f"{out_dir}/{varname1}_vs_{varname2}.png")
#     plt.close()

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

    if "C_pass_gen_matching" in good_vars:
        good_vars.remove("C_pass_gen_matching")


    category_var = "C_category"
    category_list = [1,2,3,4,5,6]

    bkg = {h:[] for h in good_vars+[category_var]}
    bkg_weight = {h:[] for h in good_vars+[category_var]}

    for bkg_tree in bkg_trees:
        data_bkg = bkg_tree.arrays(good_vars)
        weight_bkg = bkg_tree.arrays(weight_name)
        scale_factor = bkg_tree.arrays(scale_factor_vars)

        #categorize
        data_bkg = categorize_data(data_bkg, category_list, category_var)

        for h in good_vars + [category_var]:
            b_a = data_bkg[h]
            b_w = weight_bkg[weight_name]
            for sf in scale_factor_vars:
                b_w = b_w*scale_factor[sf]

            #broadcast b_w with b_a to match the shape
            b_w = ak.broadcast_arrays(b_w,b_a)[0]

            #if b_a is already flat
            if ak.Array(b_a).ndim == 1:
                bkg[h].append(ak.to_numpy(b_a))
            else:
                bkg[h].append(ak.to_numpy(ak.flatten(b_a)))
            #if b_w is already flat
            if ak.Array(b_w).ndim == 1:
                bkg_weight[h].append(ak.to_numpy(b_w))
            else:
                bkg_weight[h].append(ak.to_numpy(ak.flatten(b_w)))

            
    print("Background Variables Loaded!")
    return bkg, bkg_weight

def load_sig_data(sig_tree, good_vars, scale_factor_vars):
    category_var = "C_category"
    category_list = [1,2,3,4,5,6]

    sig = {}
    sig_weight = []

    print("Loading Signal Variables...")

    data_sig = sig_tree.arrays(good_vars)
    scale_factor = sig_tree.arrays(scale_factor_vars)

    #categorize
    data_sig = categorize_data(data_sig, category_list, category_var)

    weight_sig = np.ones_like(scale_factor[scale_factor_vars[0]])
    for sf in scale_factor_vars:
        weight_sig = weight_sig*scale_factor[sf]

    for h in good_vars + [category_var]:
        #if data_sig[h] is already flat
        if ak.Array(data_sig[h]).ndim == 1:
            sig[h] = ak.to_numpy(data_sig[h])
        else:
            sig[h] = ak.to_numpy(ak.flatten(data_sig[h]))

    #if weight_sig is already flat
    if ak.Array(weight_sig).ndim == 1:
        sig_weight = ak.to_numpy(weight_sig)
    else:
        sig_weight = ak.to_numpy(ak.flatten(weight_sig))

    print("Signal Variables Loaded!")
    return sig, sig_weight

def load_bkg_data2(bkg_trees, good_vars, weight_name, scale_factor_vars):
    if "C_pass_gen_matching" in good_vars:
        good_vars.remove("C_pass_gen_matching")

    category_var = "C_category"
    category_list = [1,2,3,4,5,6]

    bkg = []
    bkg_weight = []

    for bkg_tree in bkg_trees:
        data_bkg = bkg_tree.arrays(good_vars)
        weight_bkg = bkg_tree.arrays(weight_name)
        scale_factor = bkg_tree.arrays(scale_factor_vars)

        #categorize
        data_bkg = categorize_data(data_bkg, category_list, category_var)

        bkg_dict = {}

        # Calculate weights once
        b_w = weight_bkg[weight_name]
        for sf in scale_factor_vars:
            b_w = b_w*scale_factor[sf]

        for var in good_vars + [category_var]:
            b_a = data_bkg[var]

            #broadcast b_w with b_a to match the shape
            b_w = ak.broadcast_arrays(b_w,b_a)[0]

            #if b_a is already flat
            if ak.Array(b_a).ndim == 1:
                bkg_dict[var] = ak.to_numpy(b_a)
            else:
                bkg_dict[var] = ak.to_numpy(ak.flatten(b_a))

        #if b_w is already flat
        if ak.Array(b_w).ndim == 1:
            bkg_weight.append(ak.to_numpy(b_w))
        else:
            bkg_weight.append(ak.to_numpy(ak.flatten(b_w)))

        bkg.append(bkg_dict)

    return bkg, bkg_weight
    # bkg = {h:[] for h in good_vars+[category_var]}
    # bkg_weight = {h:[] for h in good_vars+[category_var]}

    # for bkg_tree in bkg_trees:
    #     data_bkg = bkg_tree.arrays(good_vars)
    #     weight_bkg = bkg_tree.arrays(weight_name)
    #     scale_factor = bkg_tree.arrays(scale_factor_vars)

    #     #categorize
    #     data_bkg = categorize_data(data_bkg, category_list, category_var)

    #     for h in good_vars + [category_var]:
    #         b_a = data_bkg[h]
    #         b_w = weight_bkg[weight_name]
    #         for sf in scale_factor_vars:
    #             b_w = b_w*scale_factor[sf]

    #         #broadcast b_w with b_a to match the shape
    #         b_w = ak.broadcast_arrays(b_w,b_a)[0]

    #         #if b_a is already flat
    #         if ak.Array(b_a).ndim == 1:
    #             bkg[h].append(ak.to_numpy(b_a))
    #         else:
    #             bkg[h].append(ak.to_numpy(ak.flatten(b_a)))
    #         #if b_w is already flat
    #         if ak.Array(b_w).ndim == 1:
    #             bkg_weight[h].append(ak.to_numpy(b_w))
    #         else:
    #             bkg_weight[h].append(ak.to_numpy(ak.flatten(b_w)))