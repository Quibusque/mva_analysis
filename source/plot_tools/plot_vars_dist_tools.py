import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import awkward as ak
import uproot
import mplhep as hep
import itertools

hep.style.use("CMS")

from data_tools.load_data import categorize_data

vars_name_dict = {
    "C_Ds_pt": "$p_T(D_s)$",
    "C_Ds_vertex_cos2D": "$D_s$ vertex cos(2D)",
    "C_Ds_vertex_prob": "$D_s$ vertex prob",
    "C_Hnl_vertex_2DSig_BS": "$L_{xy} / \sigma$",
    "C_Hnl_vertex_cos2D": "$N$ vertex cos(2D)",
    "C_Hnl_vertex_cos3D": "$N$ vertex cos(3D)",
    "C_Hnl_vertex_prob": "$N$ vertex prob",
    "C_mu_Ds_BS_ips_xy": "$\mu_{D}$ IPS xy",
    "C_mu_Ds_pt": "$p_T(\mu_{D})$",
    "C_mu_Ds_nValidTrackerHits": "$\mu_{D}$ tracker hits",
    "C_mu_Ds_nValidPixelHits": "$\mu_{D}$ pixel hits",
    "C_mu_Ds_tkIso_R03": "$\mu_{D}$ isolation",
    "C_mu_Hnl_BS_ips_xy": "$\mu_{H}$ IPS xy",
    "C_mu_Hnl_pt": "$p_T(\mu_{H})$",
    "C_mu_Hnl_nValidTrackerHits": "$\mu_{N}$ tracker hits",
    "C_mu_Hnl_nValidPixelHits": "$\mu_{N}$ pixel hits",
    "C_mu_Hnl_tkIso_R03": "$\mu_{N}$ isolation",
    "C_pi_BS_ip_xy": "$\pi$ IPS xy",
    "C_pi_BS_ips_xy": "$\pi$ IPS xy",
    "C_pi_pt": "$p_T(\pi)$",
    "C_pi_nValidTrackerHits": "$\pi$ tracker hits",
    "C_pi_nValidPixelHits": "$\pi$ pixel hits",
    "C_mu1mu2_dr": "$\Delta R (\mu_{H}, \mu_{D})$",
    "C_mu2pi_dr": "$\Delta R (\mu_{D}, \pi)$",
    "C_pass_gen_matching": "Pass gen matching",
    "C_mu_Hnl_charge": "$\mu_{N}$ charge",
    "C_mu_Ds_charge": "$\mu_{D}$ charge",
    "C_category": "Category",
}

vars_unit_name_dict = {
    "C_Ds_pt": "[GeV]",
    "C_Ds_vertex_cos2D": "",
    "C_Ds_vertex_prob": "",
    "C_Hnl_vertex_2DSig_BS": "",
    "C_Hnl_vertex_cos2D": "",
    "C_Hnl_vertex_cos3D": "",
    "C_Hnl_vertex_prob": "",
    "C_mu_Ds_BS_ips_xy": "",
    "C_mu_Ds_pt": "[GeV]",
    "C_mu_Ds_nValidTrackerHits": "",
    "C_mu_Ds_nValidPixelHits": "",
    "C_mu_Ds_tkIso_R03": "",
    "C_mu_Hnl_BS_ips_xy": "",
    "C_mu_Hnl_pt": "[GeV]",
    "C_mu_Hnl_nValidTrackerHits": "",
    "C_mu_Hnl_nValidPixelHits": "",
    "C_mu_Hnl_tkIso_R03": "",
    "C_pi_BS_ip_xy": "[cm]",
    "C_pi_BS_ips_xy": "",
    "C_pi_pt": "[GeV]",
    "C_pi_nValidTrackerHits": "",
    "C_pi_nValidPixelHits": "",
    "C_mu1mu2_dr": "",
    "C_mu2pi_dr": "",
    "C_pass_gen_matching": "",
    "C_mu_Hnl_charge": "",
    "C_mu_Ds_charge": "",
    "C_category": "",
}

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

bkg_label_dict = {
    "QCD_Pt-20To30": "QCD $20 < p_T < 30$",
    "QCD_Pt-30To50": "QCD $30 < p_T < 50$",
    "QCD_Pt-50To80": "QCD $50 < p_T < 80$",
    "QCD_Pt-80To120": "QCD $80 < p_T < 120$",
    "QCD_Pt-120To170": "QCD $120 < p_T < 170$",
    "QCD_Pt-170To300": "QCD $170 < p_T < 300$"
}

category_dict = {
    1: "lowDisp_SS",
    2: "mediumDisp_SS",
    3: "highDisp_SS",
    4: "lowDisp_OS",
    5: "mediumDisp_OS",
    6: "highDisp_OS"
}


# def plot_var_dist_full_bkg(
#     signals,
#     backgrounds,
#     signals_weight,
#     background_weights,
#     sig_labels,
#     bkg_labels,
#     varname,
#     out_dir,
# ):
#     # Define color and linestyle iterators
#     colors = ["black"]
#     linestyles = ["-", ":", "--", "-."]
#     color_iterator = itertools.cycle(colors)
#     linestyle_iterator = itertools.cycle(linestyles)

#     logscale = False
#     if "tkIso_R03" in varname:
#         plt.xlim(0, 60)
#         bins = np.linspace(0, 60, 100)

#     elif "nValid" in varname:
#         xmax = np.amax(signals)
#         # round xmax up to largest integer
#         xmax = int(np.ceil(xmax))
#         plt.xlim(0, xmax)
#         bins = np.linspace(0, xmax, xmax + 1)

#     elif "cos" in varname:
#         plt.xlim(-1, 1)
#         bins = np.linspace(-1, 1, 40)
#     elif "charge" in varname:
#         plt.xlim(-2, 2)
#         bins = np.linspace(-2, 2, 5)
#     elif "C_Hnl_vertex_2DSig_BS" in varname:
#         xmin = 0
#         xmax = 450
#         bins = np.linspace(xmin, xmax, 50)
#         plt.xlim(xmin, xmax)
#         logscale = True
#     else:
#         if ak.Array(signals).ndim != 1:
#             # use 1% and 99% percentile to set x limits
#             xmin = np.percentile(signals[0], 99)
#             xmax = np.percentile(signals[0], 1)
#             for signal in signals:
#                 if np.percentile(signal, 1) < xmin:
#                     xmin = np.percentile(signal, 1)
#                 if np.percentile(signal, 99) > xmax:
#                     xmax = np.percentile(signal, 99)

#             if xmin == xmax:
#                 xmin -= 1
#                 xmax += 1
#                 bins = np.linspace(xmin, xmax, 3)
#             bins = np.linspace(xmin, xmax, 50)
#             plt.xlim(xmin, xmax)
#         else:
#             xmin = np.percentile(signals, 1)
#             xmax = np.percentile(signals, 99)
#             bins = np.linspace(xmin, xmax, 50)
#             plt.xlim(xmin, xmax)

#     bkg_hists = []
#     for background, background_weight, bkg_label in zip(
#         backgrounds, background_weights, bkg_labels
#     ):
#         bkg_hist, bin_edges = np.histogram(
#             background, bins, weights=background_weight
#         )
#         bkg_hists.append(bkg_hist)
#     hep.histplot(
#         bkg_hists,
#         bins=bin_edges,
#         alpha=0.5,
#         density=True,
#         stack=True,
#         label=[bkg_label_dict[bkg_label] for bkg_label in bkg_labels],
#         histtype="fill",
#     )
#     sig_hist, bin_edges = np.histogram(signals, bins, weights=signals_weight)
#     hep.histplot(
#         sig_hist,
#         bins=bin_edges,
#         alpha=1,
#         density=True,
#         label=sig_label_dict[sig_labels],
#         histtype="step",
#         color=next(color_iterator),
#         linestyle=next(linestyle_iterator),
#         lw=2,
#     )
#     # else:
#     #     for s, s_label, s_w in zip(signals, sig_labels, signals_weight):
#     #         sig_hist, bin_edges = np.histogram(s, bins, weights=s_w)
#     #         hep.histplot(
#     #             sig_hist,
#     #             bins=bin_edges,
#     #             alpha=1,
#     #             density=True,
#     #             label=sig_label_dict[s_label],
#     #             histtype="step",
#     #             color=next(color_iterator),
#     #             linestyle=next(linestyle_iterator),
#     #             lw=2,
#     #         )

#     plt.xlabel(vars_name_dict[varname])
#     plt.ylabel("Normalized number of events")
#     plt.title(f"{vars_name_dict[varname]} Distribution")
#     plt.legend()
#     plt.savefig(f"{out_dir}/{varname}.png")
#     plt.close()


def plot_var_dist_one_sig(
    signal,
    backgrounds,
    signal_weight,
    background_weights,
    sig_labels,
    bkg_labels,
    varname,
    out_dir,
    category = None
):
    logscale = False
    if "tkIso_R03" in varname:
        plt.xlim(0, 60)
        bins = np.linspace(0, 60, 50)

    elif "nValid" in varname:
        xmax = np.amax(signal)
        # round xmax up to largest integer
        xmax = int(np.ceil(xmax))
        plt.xlim(0, xmax)
        bins = np.linspace(0, xmax, xmax + 1)

    elif "cos" in varname:
        plt.xlim(-1, 1)
        bins = np.linspace(-1, 1, 40)
    elif "charge" in varname:
        plt.xlim(-2, 2)
        bins = np.linspace(-2, 2, 5)
    elif "C_Hnl_vertex_2DSig_BS" in varname:
        xmin = np.percentile(signal, 1)
        xmax = np.percentile(signal, 90)
        bins = np.linspace(xmin, xmax, 30)
        plt.xlim(xmin, xmax)
        logscale = True
    else:
        xmin = np.percentile(signal, 1)
        xmax = np.percentile(signal, 99)
        bins = np.linspace(xmin, xmax, 50)
        plt.xlim(xmin, xmax)


    backgrounds = np.concatenate(backgrounds)
    background_weights = np.concatenate(background_weights)
    bkg_labels = ["QCD bkg"]

    bkg_hist, bin_edges = np.histogram(backgrounds, bins, weights=background_weights)
    hep.histplot(
        bkg_hist,
        bins=bin_edges,
        alpha=0.5,
        density=True,
        label=bkg_labels,
        histtype="fill",
        color="tab:blue",
    )
    sig_hist, bin_edges = np.histogram(signal, bins, weights=signal_weight)
    hep.histplot(
        sig_hist,
        bins=bin_edges,
        alpha=0.5,
        density=True,
        label=sig_label_dict[sig_labels],
        histtype="fill",
        color="tab:orange",
        lw=2,
    )
    # Set y-axis to logarithmic scale
    if logscale:
        plt.yscale('log')

    plt.xlabel(vars_name_dict[varname] + vars_unit_name_dict[varname])
    plt.ylabel("Normalized number of events")
    if category:
        plt.title(f"{vars_name_dict[varname]} Distribution {category_dict[category]}")
    else:
        plt.title(f"{vars_name_dict[varname]} Distribution")
    plt.legend()
    plt.savefig(f"{out_dir}/{varname}.png")
    plt.close()

def plot_var_dist_more_sig(
    signals,
    backgrounds,
    signals_weight,
    background_weights,
    sig_labels,
    bkg_labels,
    varname,
    out_dir,
    category = None
):
    # Define color and linestyle iterators
    linestyles = ["-", ":", "--", "-."]
    linestyle_iterator = itertools.cycle(linestyles)

    logscale = False
    if "tkIso_R03" in varname:
        plt.xlim(0, 60)
        bins = np.linspace(0, 60, 50)

    elif "nValid" in varname:
        xmax = np.amax(signals)
        # round xmax up to largest integer
        xmax = int(np.ceil(xmax))
        plt.xlim(0, xmax)
        bins = np.linspace(0, xmax, xmax + 1)

    elif "cos" in varname:
        plt.xlim(-1, 1)
        bins = np.linspace(-1, 1, 40)
    elif "charge" in varname:
        plt.xlim(-2, 2)
        bins = np.linspace(-2, 2, 5)
    elif "C_Hnl_vertex_2DSig_BS" in varname:
        xmin = np.percentile(signals[0], 90)
        xmax = -1
        for signal in signals:
            if np.percentile(signal, 1) < xmin:
                xmin = np.percentile(signal, 1)
            if np.percentile(signal, 99) > xmax:
                xmax = np.percentile(signal, 99)

        if xmin == xmax:
            xmin -= 1
            xmax += 1
            bins = np.linspace(xmin, xmax, 3)
        bins = np.linspace(xmin, xmax, 40)
        plt.xlim(xmin, xmax)
        logscale = True
    else:
        # use 1% and 99% percentile to set x limits
        xmin = np.percentile(signals[0], 99)
        xmax = np.percentile(signals[0], 1)
        for signal in signals:
            if np.percentile(signal, 1) < xmin:
                xmin = np.percentile(signal, 1)
            if np.percentile(signal, 99) > xmax:
                xmax = np.percentile(signal, 99)

        if xmin == xmax:
            xmin -= 1
            xmax += 1
            bins = np.linspace(xmin, xmax, 3)
        bins = np.linspace(xmin, xmax, 50)
        plt.xlim(xmin, xmax)


    backgrounds = np.concatenate(backgrounds)
    background_weights = np.concatenate(background_weights)
    bkg_labels = ["QCD bkg"]

    bkg_hist, bin_edges = np.histogram(backgrounds, bins, weights=background_weights)
    hep.histplot(
        bkg_hist,
        bins=bin_edges,
        alpha=0.5,
        density=True,
        label=bkg_labels,
        histtype="fill",
    )

    for s, s_label, s_w in zip(signals, sig_labels, signals_weight):
        sig_hist, bin_edges = np.histogram(s, bins, weights=s_w)
        hep.histplot(
            sig_hist,
            bins=bin_edges,
            alpha=1,
            density=True,
            label=sig_label_dict[s_label],
            histtype="step",
            color="black",
            linestyle=next(linestyle_iterator),
            lw=2,
        )
    
    # Set y-axis to logarithmic scale
    if logscale:
        plt.yscale('log')

    plt.xlabel(vars_name_dict[varname] + vars_unit_name_dict[varname])
    plt.ylabel("Normalized number of events")
    if category:
        plt.title(f"{vars_name_dict[varname]} Distribution {category_dict[category]}")
    else:
        plt.title(f"{vars_name_dict[varname]} Distribution")
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


def load_bkg_data(bkg_trees, good_vars, weight_name, scale_factor_vars):
    if "C_pass_gen_matching" in good_vars:
        good_vars.remove("C_pass_gen_matching")

    category_var = "C_category"
    category_list = [1, 2, 3, 4, 5, 6]

    bkg = {h: [] for h in good_vars + [category_var]}
    bkg_weight = {h: [] for h in good_vars + [category_var]}

    for bkg_tree in bkg_trees:
        data_bkg = bkg_tree.arrays(good_vars)
        weight_bkg = bkg_tree.arrays(weight_name)
        scale_factor = bkg_tree.arrays(scale_factor_vars)

        # categorize
        data_bkg = categorize_data(data_bkg, category_list, category_var)

        for h in good_vars + [category_var]:
            b_a = data_bkg[h]
            b_w = weight_bkg[weight_name]
            for sf in scale_factor_vars:
                b_w = b_w * scale_factor[sf]

            # broadcast b_w with b_a to match the shape
            b_w = ak.broadcast_arrays(b_w, b_a)[0]

            # if b_a is already flat
            if ak.Array(b_a).ndim == 1:
                bkg[h].append(ak.to_numpy(b_a))
            else:
                bkg[h].append(ak.to_numpy(ak.flatten(b_a)))
            # if b_w is already flat
            if ak.Array(b_w).ndim == 1:
                bkg_weight[h].append(ak.to_numpy(b_w))
            else:
                bkg_weight[h].append(ak.to_numpy(ak.flatten(b_w)))

    print("Background Variables Loaded!")
    return bkg, bkg_weight


def load_sig_data(sig_tree, good_vars, scale_factor_vars):
    category_var = "C_category"
    category_list = [1, 2, 3, 4, 5, 6]

    sig = {}
    sig_weight = []

    print("Loading Signal Variables...")

    data_sig = sig_tree.arrays(good_vars)
    scale_factor = sig_tree.arrays(scale_factor_vars)

    # categorize
    data_sig = categorize_data(data_sig, category_list, category_var)

    weight_sig = np.ones_like(scale_factor[scale_factor_vars[0]])
    for sf in scale_factor_vars:
        weight_sig = weight_sig * scale_factor[sf]

    for h in good_vars + [category_var]:
        # if data_sig[h] is already flat
        if ak.Array(data_sig[h]).ndim == 1:
            sig[h] = ak.to_numpy(data_sig[h])
        else:
            sig[h] = ak.to_numpy(ak.flatten(data_sig[h]))

    # if weight_sig is already flat
    if ak.Array(weight_sig).ndim == 1:
        sig_weight = ak.to_numpy(weight_sig)
    else:
        sig_weight = ak.to_numpy(ak.flatten(weight_sig))

    print("Signal Variables Loaded!")
    return sig, sig_weight


def load_bkg_data(bkg_trees, good_vars, weight_name, scale_factor_vars):
    if "C_pass_gen_matching" in good_vars:
        good_vars.remove("C_pass_gen_matching")

    category_var = "C_category"
    category_list = [1, 2, 3, 4, 5, 6]

    bkg = []
    bkg_weight = []

    for bkg_tree in bkg_trees:
        data_bkg = bkg_tree.arrays(good_vars)
        weight_bkg = bkg_tree.arrays(weight_name)
        scale_factor = bkg_tree.arrays(scale_factor_vars)

        # categorize
        data_bkg = categorize_data(data_bkg, category_list, category_var)

        bkg_dict = {}

        # Calculate weights once
        b_w = weight_bkg[weight_name]
        for sf in scale_factor_vars:
            b_w = b_w * scale_factor[sf]

        for var in good_vars + [category_var]:
            b_a = data_bkg[var]

            # broadcast b_w with b_a to match the shape
            b_w = ak.broadcast_arrays(b_w, b_a)[0]

            # if b_a is already flat
            if ak.Array(b_a).ndim == 1:
                bkg_dict[var] = ak.to_numpy(b_a)
            else:
                bkg_dict[var] = ak.to_numpy(ak.flatten(b_a))

        # if b_w is already flat
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
