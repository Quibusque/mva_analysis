import numpy as np
import pandas as pd

main_dir = "results_categories/myMVA"
label_list = ["mN1p0_ctau10"]

top_variables = []
threshold = 0.05

# Create a dictionary to store the highest importance value for each variable
variable_importance = {}

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

        # Update the highest importance value for each variable
        for var in vars:
            importance = df[df["variables"] == var][f"importance_{label}"].max()
            if var not in variable_importance or importance > variable_importance[var][0]:
                variable_importance[var] = (importance, label, i)

top_variables = list(set(top_variables))

print(f"There are {len(top_variables)} variables with importance above {threshold}:")
for var in top_variables:
    print(f"{var} with highest importance of {variable_importance[var][0]} for label {variable_importance[var][1]} in category {variable_importance[var][2]}")