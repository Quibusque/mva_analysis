import numpy as np
import pandas as pd

main_dir = "results_categories_test/myMVA"
label_list = ["mN1p0_ctau10", "mN1p25_ctau10", "mN1p5_ctau10", "mN1p8_ctau10"]
n_top_variables = 4

full_vars = []
for label in label_list:
    input = f"{main_dir}/{label}"
    for cat1,cat2 in ([1,4],[2,5],[3,6]):
        category_input1 = input + "/cat_" + str(cat1)
        category_input2 = input + "/cat_" + str(cat2)
        #read importance.csv in category_input dir
        #first line is column names
        df1 = pd.read_csv(category_input1 + "/importance.csv")
        df2 = pd.read_csv(category_input2 + "/importance.csv")
        top_variables1 = df1["variables"].tolist()[0:n_top_variables]
        top_variables2 = df2["variables"].tolist()[0:n_top_variables]
        vars = list(set(top_variables1 + top_variables2))
        #check that the position of vars in top_variables1 and top_variables2 is the same
        #or at most one position off
        for var in vars:
            if var in top_variables1 and var in top_variables2:
                pos1 = df1["variables"].tolist().index(var)
                pos2 = df2["variables"].tolist().index(var)
                if abs(pos1-pos2) > 1:
                    print(f"Variable {var} has position {pos1} in cat{cat1} and {pos2} in cat{cat2} in mass hyp {label}")
        
    full_vars.extend(vars)
    full_vars = list(set(full_vars))
print(f"There are {len(full_vars)} variables in the top {n_top_variables} for all mass hypotheses and categories:")
for var in full_vars:
    print(var)
# top_variables = []
# threshold = 0.05

# # Create a dictionary to store the highest importance value for each variable
# variable_importance = {}

# for label in label_list:
#     input = f"{main_dir}/{label}"
#     for i in range(1, 7):
#         category_input = input + "/cat_" + str(i)
#         #read importance.csv in category_input dir
#         #first line is column names
#         df = pd.read_csv(category_input + "/importance.csv")
#         #importance_label must be > threshold
#         df = df[df[f"importance_{label}"] > threshold]
#         vars = df["variables"].tolist()
#         top_variables.extend(vars)

#         # Update the highest importance value for each variable
#         for var in vars:
#             importance = df[df["variables"] == var][f"importance_{label}"].max()
#             if var not in variable_importance or importance > variable_importance[var][0]:
#                 variable_importance[var] = (importance, label, i)

# top_variables = list(set(top_variables))

# print(f"There are {len(top_variables)} variables with importance above {threshold}:")
# for var in top_variables:
#     print(f"{var} with highest importance of {variable_importance[var][0]} for label {variable_importance[var][1]} in category {variable_importance[var][2]}")