#!/bin/bash



OUT_DIR=results_categories_adaboost
mkdir -p $OUT_DIR

#mass list ("mN1p5" "mN1p0")
mass_list=("mN1p0")

for mass in "${mass_list[@]}"
do
    python3 source/training_my_mva_categories.py  --train --results --plots  --mass $mass --out_dir $OUT_DIR 
done    