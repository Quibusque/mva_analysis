from add_score_tools.add_score_tools import (
    append_multiple_scores_overwrite,
    get_multiple_bdt_outputs,
    write_multiple_scores_root,
)
from cfg.hnl_mva_tools import read_json_file
from mva_tools.mva_training_tools import load_model
import uproot
import os
import re
import awkward as ak
import numpy as np

input_files_list = [
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_DsToNMu_NToMuPi_SoftQCDnonD_noQuarkFilter_mN1p0_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_DsToNMu_NToMuPi_SoftQCDnonD_noQuarkFilter_mN1p25_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_DsToNMu_NToMuPi_SoftQCDnonD_noQuarkFilter_mN1p5_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_DsToNMu_NToMuPi_SoftQCDnonD_noQuarkFilter_mN1p8_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen_tree.root"
    # add comma
    # "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_ParkingBPH1_Run2018A-UL2018_MiniAODv2-v1_tree.root",
    # "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    # "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    # "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    # "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    # "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    # "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
]

category_list = [1, 2, 3, 4, 5, 6]
category_var = "C_category"

trained_models_dirs = [
    "results_categories/myMVA/mN1p0_ctau10",
    "results_categories/myMVA/mN1p25_ctau10",
    "results_categories/myMVA/mN1p5_ctau10",
    "results_categories/myMVA/mN1p8_ctau10",
]

ntuples_json = "source/cfg/ntuples.json"
vars_json = "source/cfg/vars_new.json"
my_method = "XGBoost"

ntuples = read_json_file(ntuples_json)
treename = ntuples["treename"]


good_vars = read_json_file(vars_json)["vars"]
training_vars = read_json_file(vars_json)["training_vars"]


# ┌─────────────────────────────┐
# │ LOAD ALL THE XGBOOST MODELS │
# └─────────────────────────────┘
mass_hypotheses = []
xgboost_models_dict = {}
for trained_model_dir in trained_models_dirs:
    # check that dir exists
    assert os.path.isdir(trained_model_dir)
    mass_hyp = re.findall(r"(mN1p\d+_ctau\d+)", trained_model_dir)[0]
    mass_hypotheses.append(mass_hyp)
    xgboost_models_dict[mass_hyp] = []
    for category in category_list:
        category_dir = f"{trained_model_dir}/cat_{category}"
        print(f"Loading {my_method} model from {category_dir}")
        model = load_model(f"{category_dir}/{my_method}_model", my_method)
        xgboost_models_dict[mass_hyp].append(model)
print(f"Loaded {my_method} models")

for input_file in input_files_list:
    print(f"Processing {input_file}")
    # open the tree with uproot
    with uproot.open(input_file) as f:
        my_tree = f[treename]
        data_dict = my_tree.arrays(good_vars, library="ak", how=dict)
        # get the bdt output
        bdt_outputs = get_multiple_bdt_outputs(
            data_dict, training_vars, category_list, xgboost_models_dict
        )
        assert len(bdt_outputs) == len(mass_hypotheses)

        print(f"n_entries from bdt_outputs: {len(bdt_outputs[0])}")
        print(f"n_entries from data_dict: {len(data_dict['C_Ds_pt'])}")
        print(f"n_entries from my_tree: {len(my_tree['C_Ds_pt'])}")
    # my_tree = uproot.open(input_file)[treename]

    # # get the data from the tree
    # data_dict = my_tree.arrays(good_vars, library="ak", how=dict)
    # # get the bdt output
    # bdt_outputs = get_multiple_bdt_outputs(
    #     data_dict, training_vars, category_list, xgboost_models_dict
    # )
    # assert len(bdt_outputs) == len(mass_hypotheses)

    break
    # for bdt_output, mass_hyp in zip(bdt_outputs, mass_hypotheses):
    #     #flatten it
    #     bdt_output = ak.flatten(bdt_output)
    #     #make histogram
    #     plt.hist(bdt_output, bins=100)
    #     plt.xlabel(f"BDT output for {mass_hyp}")
    #     plt.ylabel("Events")
    #     out_dir = "test_plots"
    #     if not os.path.isdir(out_dir):
    #         os.makedirs(out_dir)
    #     plt.savefig(f"{out_dir}/{mass_hyp}.png")

    # append the bdt output to the tree
    score_name_list = [
        "C_" + my_method.lower() + "_score_" + mass_hyp for mass_hyp in mass_hypotheses
    ]
    output_file = input_file.replace(".root", "_with_scores.root")
    write_multiple_scores_root(
        input_file, output_file, treename, bdt_outputs, score_name_list
    )

    # open the output file with uproot
    my_new_tree = uproot.open(output_file)[treename]
    # get the data from the tree
    bdt_outputs_new_dict = my_new_tree.arrays(score_name_list, library="ak", how=dict)
    bdt_outputs_new = [bdt_outputs_new_dict[score_name] for score_name in score_name_list]

    # check that the bdt outputs are the same
    for main_index, (bdt_output, bdt_output_new) in enumerate(zip(bdt_outputs, bdt_outputs_new)):
        print(f"main_index: {main_index}")
        if ak.all(bdt_output == bdt_output_new):
            print("All good")
            continue
        #print index where they are different, keep in mind its awkward
        for index, (el1,el2) in enumerate(zip(bdt_output, bdt_output_new)):
            if ak.any(el1 != el2):
                print(f"index: {index}")
                print(f"el1: {el1}")
                print(f"el2: {el2}")
                break

    break

# # check that dir exists
# assert os.path.isdir(trained_model_dir)
# xgboost_models = []
# for category in category_list:
#     category_dir = f"{trained_model_dir}/cat_{category}"
#     model = load_model(f"{category_dir}/{my_method}_model", my_method)
#     xgboost_models.append(model)
