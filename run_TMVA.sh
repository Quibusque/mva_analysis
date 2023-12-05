#!/bin/bash


OUT_DIR=results_vars_new
mkdir -p $OUT_DIR

#mass list ("mN1p5" "mN1p0")
mass_list=("mN1p0" "mN1p5")

for mass in "${mass_list[@]}"
do
    python3 source/training_TMVA.py --mass $mass --out_dir $OUT_DIR --new| tee $OUT_DIR/tmva_log.txt
    #this is to remove the progress bar from the log file
    grep -v "ETA\|=>\.|[==============================]" $OUT_DIR/tmva_log.txt > $OUT_DIR/tmva_log_clean.txt
done

