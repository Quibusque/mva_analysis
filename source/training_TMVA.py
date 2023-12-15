from ROOT import TMVA, TFile, TTree, TCut, TChain
from subprocess import call
import os
import re
import argparse

from cfg.hnl_mva_tools import read_json_file


def filter_trees(trees, tree_labels, mass_list, ctau_list):
    """
    example:
    mass_list = ["mN1p0", "mN1p5"]
    ctau_list = ["ctau100", "ctau10"]
    """
    assert isinstance(mass_list[0], str)
    assert isinstance(ctau_list[0], str)

    return_trees = []
    return_labels = []
    for tree, label in zip(trees, tree_labels):
        mn = re.findall(r"(mN\dp\d+)", label)[0]
        ct = re.findall(r"(ctau\d+)", label)[0]
        if mn in mass_list and ct in ctau_list:
            return_trees.append(tree)
            return_labels.append(label)

    return return_trees, return_labels


# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()


# if name is main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass", type=str, help="single mass label of form mN1p0", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--new", action="store_true", help="use new vars")
    group.add_argument("--old", action="store_true", help="use old vars")
    parser.add_argument("--transf", action="store_true", help="transform variables")
    parser.add_argument("--out_dir", type=str, help="main output directory for all", required=True)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir = f"{out_dir}/TMVA"

    ntuples = read_json_file("source/cfg/ntuples.json")
    signal_file_names = ntuples["signal"]
    background_file_names = ntuples["background"]
    treename = ntuples["treename"]
    weight_name = ntuples["weight_name"]

    # labels for signal and background
    sig_labels = [re.findall(r"(mN1p\d+_ctau\d+)", f)[0] for f in signal_file_names]
    bkg_labels = [
        s[s.rfind("QCD_Pt-") : s.rfind("_MuEnriched")] for s in background_file_names
    ]
    # list of variables to use, from cfg file
    if args.old:
        vars_json = "source/cfg/vars_old.json"
        good_vars = read_json_file(vars_json)["vars"]
        if "C_pass_gen_matching" in good_vars:
            good_vars.remove("C_pass_gen_matching")
    elif args.new:
        vars_json = "source/cfg/vars_new.json"
        good_vars = read_json_file(vars_json)["training_vars"]
        if "C_pass_gen_matching" in good_vars:
            good_vars.remove("C_pass_gen_matching")


    # hyperparameters settings
    hyperpars = read_json_file("source/cfg/hyperparameters.json")

    # test fraction
    test_fraction = 0.2

    # ┌──────────────────────────┐
    # │ CHOOSE YOUR SIGNAL FILES │
    # └──────────────────────────┘
    mass_list = [args.mass]
    ctau_list = ["ctau10"]
    my_sig_trees, my_sig_labels = filter_trees(
        signal_file_names, sig_labels, mass_list, ctau_list
    )

    for signal_label, signal_file_name in zip(my_sig_labels, my_sig_trees):
        print(f"running TMVA for signal {signal_label}")
        # results directory
        results_dir = f"{out_dir}/{signal_label}"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # ┌────────────────────────────────────┐
        # │ HYPERPARAMETERS SETTINGS FOR KERAS │
        # └────────────────────────────────────┘
        keras_par_string = "H:!V:"

        if args.transf:
            keras_par_string += "VarTransform=D,G:"
        if args.new:
            keras_filename_model = hyperpars["keras_shallow"]["filename_model_new"]
            keras_filename_trained_model = hyperpars["keras_shallow"][
            "filename_trained_model_new"
        ]
        elif args.old:
            keras_filename_model = hyperpars["keras_shallow"]["filename_model_old"]
            keras_filename_trained_model = hyperpars["keras_shallow"][
            "filename_trained_model_old"
        ]

        keras_num_epochs = hyperpars["keras_shallow"]["epochs"]
        keras_batch_size = hyperpars["keras_shallow"]["batch_size"]
        keras_verbose = 1
        keras_par_string += f"FilenameModel={keras_filename_model}:FilenameTrainedModel={keras_filename_trained_model}:NumEpochs={keras_num_epochs}:BatchSize={keras_batch_size}:Verbose={keras_verbose}"

        print(f"keras_par_string: {keras_par_string}")

        # ┌──────────────────────────────────┐
        # │ HYPERPARAMETERS SETTINGS FOR BDT │
        # └──────────────────────────────────┘
        BDT_par_string = "!H:!V:"
        BDT_ntrees = hyperpars["adaboost"]["n_estimators"]
        BDT_min_node_size = hyperpars["adaboost"]["min_samples_leaf"]
        BDT_max_depth = hyperpars["adaboost"]["max_depth"]
        BDT_par_string += f"NTrees={BDT_ntrees}:MinNodeSize={BDT_min_node_size}:MaxDepth={BDT_max_depth}"

        print(f"dealing with signal {signal_label}")
        print(f"signal file name: {signal_file_name}")

        # # REMOVE NAN VALUES THAT VERTEX VARS CAN HAVE
        # vertex_vars = [var for var in good_vars if re.search("vertex", var)]
        # cut_string = ""
        # for var in vertex_vars:
        #     cut_string += f"!TMath::IsNaN({var}) && "
        # cut_string = cut_string[:-4]


        output = TFile.Open(f"{results_dir}/TMVA_output.root", "RECREATE")

        if args.transf:
            factory_string = "!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=Classification"
        else:
            factory_string = (
                "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification"
            )

        print(f"factory_string: {factory_string}")
        factory = TMVA.Factory(
            "TMVAClassification",
            output,
            factory_string,
        )

        # pyROOT setup for background and signal
        bkg_files = [TFile(f) for f in background_file_names]
        bkg_trees = [f.Get(treename) for f in bkg_files]

        signal_file = TFile(signal_file_name)

        # pyROOT dataloader
        dataloader = TMVA.DataLoader("dataset")

        for var in good_vars:
            dataloader.AddVariable(var)

        # use TChain for multiple files
        background = TChain(treename)
        for bkg in background_file_names:
            background.Add(bkg)

        signal_file = TFile(signal_file_name)
        signal = signal_file.Get(treename)

        # prepare dataloader
        dataloader.AddSignalTree(signal, 1.0)
        dataloader.AddBackgroundTree(background, 1.0)

        # splitting entries for training and testing
        n_signal = signal.GetEntries()
        n_background = background.GetEntries()

        test_sgn = int(n_signal * test_fraction)
        test_bkg = int(n_background * test_fraction)
        train_sgn = n_signal - test_sgn
        train_bkg = n_background - test_bkg

        # NormMode=EqualNumEvents, important
        dataloader.PrepareTrainingAndTestTree(
            TCut(""),
            f"nTrain_Signal={train_sgn}:nTrain_Background={train_bkg}:nTest_Signal={test_sgn}:nTest_Background={test_bkg}:SplitMode=Random:NormMode=EqualNumEvents:!V",
        )
        # dataloader.SetSignalWeightExpression(weight_name)
        dataloader.SetBackgroundWeightExpression(weight_name)

        # Book methods
        factory.BookMethod(
            dataloader,
            TMVA.Types.kPyKeras,
            "PyKeras",
            keras_par_string,
        )
        #book tmva BDT
        factory.BookMethod(
            dataloader,
            TMVA.Types.kBDT,
            "BDT",
            BDT_par_string,
        )

        # Run training, test and evaluation
        factory.TrainAllMethods()
        factory.TestAllMethods()
        factory.EvaluateAllMethods()
