#!/bin/bash

INPUT_DIR=results_vars_new

#make output dir inside input dir, if it doesn't exist
OUT_DIR=$INPUT_DIR/plots
mkdir -p $OUT_DIR
python3 source/make_plots_from_results.py --tmva --mymva --input_dir $INPUT_DIR --input_dir_tmva  $INPUT_DIR  --out_dir $OUT_DIR
