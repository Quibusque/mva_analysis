import numpy as np
import pandas as pd

main_dir = "results_categories/myMVA"
label_list = ["mN1p0_ctau10"]

top_variables = []
threshold = 0.1

for label in label_list:
    input = f"{main_dir}/{label}"
    for i in range(1, 7):
        category_input = input + "/cat_" + str(i)
        #read importance.csv in category_input dir
        #first line is column names
        df = pd.read_csv(category_input + "/importance.csv")
        #importance_label must be > threshold
        df = df[df[f"importance_{label}"] > threshold]
        vars = df["variables"].tolist()
        top_variables.extend(vars)

top_variables = list(set(top_variables))

print(f"There are {len(top_variables)} variables with importance above {threshold}:")
for var in top_variables:
    print(var)