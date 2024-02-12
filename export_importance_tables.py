import numpy as np
import pandas as pd
import os

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

category_dict = {
    1: "lowDisp\_SS",
    2: "mediumDisp\_SS",
    3: "highDisp\_SS",
    4: "lowDisp\_OS",
    5: "mediumDisp\_OS",
    6: "highDisp\_OS"
}

def make_latex_table(df: pd.DataFrame) -> str:


    # Start the table and tabular environment
    latex_table = "\\begin{table}[]\n\\begin{tabular}{c|c}\n"


    # Add the column names
    latex_table += f"{df.columns[0]} & {df.columns[1]} \\\\ \\hline\n"

    # Add the rows
    for index, row in df.iterrows():
        latex_table += f"{row[0]} & {row[1]} \\\\\n"

    # End the tabular and table environment
    latex_table += "\\end{tabular}\n\\end{table}"

    return latex_table


def generate_latex_tables(dfs):
    latex_table = "\\begin{table}[htb]\n  \\centering\n"

    # \begin{adjustbox}{minipage=1.1\textwidth,center}
    latex_table += "  \\begin{adjustbox}{minipage=1.1\\textwidth,center}\n"
    for i, df in enumerate(dfs, start=1):
        latex_table += "  \\subfloat[Category {}]{{\n".format(category_dict[i])
        latex_table += "    \\begin{tabular}{c|c}\n"
        latex_table += "      {}              & {}  \\\\ \\hline\n".format("variable","importance")
        for _, row in df.iterrows():
            latex_table += "      {}    & {}   \\\\\n".format(vars_name_dict[row[0]], row[1])
        latex_table += "    \\end{tabular}}\n"
        latex_table += "  \\hfill\n"
    latex_table += "  \\end{adjustbox}\n"
    latex_table += "\\end{table}"
    return latex_table

main_dir = "results_categories_test/myMVA"
label_list = ["mN1p0_ctau10", "mN1p25_ctau10", "mN1p5_ctau10", "mN1p8_ctau10"]
n_top_variables = 4
out_dir = "latex_tables"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)



for label in label_list:
    input = f"{main_dir}/{label}"
    dfs = []
    for cat in [1,2,3,4,5,6]:
        category_input = f"{input}/cat_{cat}"
        df = pd.read_csv(category_input + "/importance.csv")
        top_df = df.head(n_top_variables)

        #round the importance values to 2 decimal places
        top_df[f"importance_{label}"] = top_df[f"importance_{label}"].apply(lambda x: round(x, 4))
        dfs.append(top_df)

    latex_tables = generate_latex_tables(dfs)
    with open(f"{out_dir}/{label}.txt", "w") as f:
        f.write(latex_tables)





# import numpy as np
# import pandas as pd

# main_dir = "results_categories_test/myMVA"
# label_list = ["mN1p0_ctau10", "mN1p25_ctau10", "mN1p5_ctau10", "mN1p8_ctau10"]
# n_top_variables = 4

# full_vars = []
# for label in label_list:
#     input = f"{main_dir}/{label}"
#     for cat1,cat2 in ([1,4],[2,5],[3,6]):
#         category_input1 = input + "/cat_" + str(cat1)
#         category_input2 = input + "/cat_" + str(cat2)
#         #read importance.csv in category_input dir
#         #first line is column names
#         df1 = pd.read_csv(category_input1 + "/importance.csv")
#         df2 = pd.read_csv(category_input2 + "/importance.csv")
#         top_variables1 = df1["variables"].tolist()[0:n_top_variables]
#         top_variables2 = df2["variables"].tolist()[0:n_top_variables]
#         vars = list(set(top_variables1 + top_variables2))
#         #check that the position of vars in top_variables1 and top_variables2 is the same
#         #or at most one position off
#         for var in vars:
#             if var in top_variables1 and var in top_variables2:
#                 pos1 = df1["variables"].tolist().index(var)
#                 pos2 = df2["variables"].tolist().index(var)
#                 if abs(pos1-pos2) > 1:
#                     print(f"Variable {var} has position {pos1} in cat{cat1} and {pos2} in cat{cat2} in mass hyp {label}")
        
#     full_vars.extend(vars)
#     full_vars = list(set(full_vars))
# print(f"There are {len(full_vars)} variables in the top {n_top_variables} for all mass hypotheses and categories:")
# for var in full_vars:
#     print(var)
# # top_variables = []
# # threshold = 0.05

# # # Create a dictionary to store the highest importance value for each variable
# # variable_importance = {}

# # for label in label_list:
# #     input = f"{main_dir}/{label}"
# #     for i in range(1, 7):