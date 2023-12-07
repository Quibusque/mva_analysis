#!/bin/bash



OUT_DIR=results_vars_new
mkdir -p $OUT_DIR

#mass list ("mN1p5" "mN1p0")
mass_list=("mN1p0")

for mass in "${mass_list[@]}"
do
    python3 source/training_my_mva.py --train --results --old --mass $mass --out_dir $OUT_DIR #| tee $OUT_DIR/mymva_log.txt
    #this is to remove the progress bar from the log file
    #grep -v "ETA\|=>\.|[==============================]" $OUT_DIR/mymva_log.txt > $OUT_DIR/mymva_log_clean.txt
done    