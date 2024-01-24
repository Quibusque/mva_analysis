from add_score_tools.add_score_tools import get_multiple_bdt_outputs, write_score_root
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
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_DsToNMu_NToMuPi_SoftQCDnonD_noQuarkFilter_mN1p8_ctau10p0mm_TuneCP5_13TeV-pythia8-evtgen_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_ParkingBPH1_Run2018A-UL2018_MiniAODv2-v1_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-120To170_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-170To300_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-20To30_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-30To50_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-50To80_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
    "/home/quibus/hnl_ntuples_for_mva/tree_HnlToMuPi_prompt_QCD_Pt-80To120_MuEnrichedPt5_TuneCP5_13TeV-pythia8_tree.root",
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
    assert os.path.isfile(input_file)
    # event number is used later to cross check, make sure it is in the list
    # of variables to load. C_pass_gen_matching is not available in data
    if "event" not in good_vars:
        good_vars.append("event")
    if "C_pass_gen_matching" in good_vars:
        good_vars.remove("C_pass_gen_matching")

    
    data_dict = uproot.open(input_file)[treename].arrays(
        library="ak", how=dict, expressions=good_vars
    )
    old_event_array = data_dict["event"]

    # ┌────────────────────┐
    # │ COMPUTE BDT SCORES │
    # └────────────────────┘

    bdt_outputs, category_array, event_array = get_multiple_bdt_outputs(
        data_dict,
        training_vars=training_vars,
        category_list=category_list,
        xgboost_models_dict=xgboost_models_dict,
    )

    assert ak.all(old_event_array == event_array)

    # ┌───────────────┐
    # │ WRITE TO FILE │
    # └───────────────┘

    # create output file name as input file name with scores_ prepended
    output_file = os.path.join(
        os.path.dirname(input_file), "scores_" + os.path.basename(input_file)
    )
    # create list of score_names
    score_names = [f"C_{(my_method).lower()}_{mass_hyp}" for mass_hyp in mass_hypotheses]

    # write the scores to the output file
    write_score_root(
        input_file,
        output_file,
        treename,
        bdt_outputs,
        score_names,
        event_array,
    )
