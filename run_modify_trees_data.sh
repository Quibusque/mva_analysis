#!/bin/bash



MAIN_DIR="/home/quibus/hnl_ntuples_for_mva/"

INPUT=$MAIN_DIR"tree_HnlToMuPi_prompt_ParkingBPH1_Run2018A-UL2018_MiniAODv2-v1_tree.root"
OUTPUT=$MAIN_DIR"tree_HnlToMuPi_prompt_ParkingBPH1_Run2018A-UL2018_MiniAODv2-v1_tree_modified.root"

python3 source/modify_ttrees_data.py --input $INPUT --output $OUTPUT