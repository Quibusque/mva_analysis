#!/bin/bash

#SCAN 1
# "XGBoost": {
#     "max_depth": 3,
#     "n_estimators": 30,
#     "early_stopping_rounds": 10
# },
#SCAN2
# "XGBoost": {
#     "max_depth": 3,
#     "n_estimators": 20,
#     "early_stopping_rounds": 5
# },

#mass list ("mN1p5" "mN1p0")
mass_list=("mN1p0" "mN1p5")


#add binary objectives only
objectives=("binary:hinge" "binary:logistic")
eval_metrics=("rmse" "mae" "rmsle" "error")
OUT_DIR="output_mymva_scan2"
mkdir -p $OUT_DIR

for mass in "${mass_list[@]}"
do
    python3 source/training_my_mva_categories_scan.py --train --results --plots --eval_metric "${eval_metrics[@]}" --objective "${objectives[@]}" --mass $mass --out_dir $OUT_DIR > ${OUT_DIR}/log_${mass}.txt
done