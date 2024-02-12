import numpy as np
import pandas as pd

vars_name_dict = {
    "C_Ds_pt":"$p_T(D_s)$",
    "C_Ds_vertex_cos2D":"$\cos\\theta(2D) (D_s)$",
    "C_Ds_vertex_prob":"$D_s$ vertex prob",
    "C_Hnl_vertex_2DSig_BS":"$L_{xy} / \sigma$",
    "C_Hnl_vertex_cos2D":"$\cos\\theta(2D) (N)$",
    "C_Hnl_vertex_cos3D":"$\cos\\theta(3D) (N)$",
    "C_Hnl_vertex_prob":"$N$ vertex prob",
    "C_mu_Ds_BS_ips_xy":"$\mu_{D}$ IPS xy",
    "C_mu_Ds_pt":"$p_T(\mu_{D})$",
    "C_mu_Ds_nValidTrackerHits":"$\mu_{D}$ tracker hits",
    "C_mu_Ds_nValidPixelHits":"$\mu_{D}$ pixel hits",
    "C_mu_Ds_tkIso_R03":"$\mu_{D}$ isolation",
    "C_mu_Hnl_BS_ips_xy":"$\mu_{H}$ IPS xy",
    "C_mu_Hnl_pt":"$p_T(\mu_{H})$",
    "C_mu_Hnl_nValidTrackerHits":"$\mu_{N}$ tracker hits",
    "C_mu_Hnl_nValidPixelHits":"$\mu_{N}$ pixel hits",
    "C_mu_Hnl_tkIso_R03":"$\mu_{N}$ isolation",
    "C_pi_BS_ip_xy":"$\pi$ IPS xy",
    "C_pi_BS_ips_xy":"$\pi$ IPS xy",
    "C_pi_pt":"$p_T(\pi)$",
    "C_pi_nValidTrackerHits":"$\pi$ tracker hits",
    "C_pi_nValidPixelHits":"$\pi$ pixel hits",
    "C_mu1mu2_dr":"$\Delta R (\mu_{H}, \mu_{D})$",
    "C_mu2pi_dr":"$\Delta R (\mu_{D}, \pi)$",
    "C_pass_gen_matching":"Pass gen matching",
    "C_mu_Hnl_charge":"$\mu_{N}$ charge",
    "C_mu_Ds_charge":"$\mu_{D}$ charge"
}

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
    print(vars_name_dict[var])
# top_variables = []
# threshold = 0.05
#┌───────────────────────┐
#│ VAR IMPORTANCE TABLES │
#└───────────────────────┘
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