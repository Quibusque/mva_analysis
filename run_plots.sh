#!/bin/bash

INPUT_DIR=results_vars_old

#make output dir inside input dir, if it doesn't exist
OUT_DIR=$INPUT_DIR/plots
mkdir -p $OUT_DIR
python3 source/make_plots_from_results.py  --mymva --input_dir $INPUT_DIR  --out_dir $OUT_DIR
