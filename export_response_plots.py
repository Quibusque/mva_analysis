import os 
import numpy as np
import pandas as pd
import shutil

main_dir = "results_categories/myMVA"
label_list = ["mN1p0_ctau10","mN1p5_ctau10"]
out_dir = "pictures_for_thesis"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#if there is already something in the out_dir, delete it first
for file in os.listdir(out_dir):
    file_path = os.path.join(out_dir, file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)

for label in label_list:
    input = f"{main_dir}/{label}"
    dfs = []
    for cat in [1,2,3,4,5,6]:
        category_input = f"{input}/cat_{cat}/plots"
        xgb_response_pic = f"{category_input}/XGBoost_response.png"
        #copy this to out_dir
        destionation = f"{out_dir}/{label}_cat{cat}_XGB_response.png"
        shutil.copy(xgb_response_pic, destionation)
        