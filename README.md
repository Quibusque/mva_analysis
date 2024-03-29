## How to run

### Before running scripts
#### pre-operations on TTrees

Modify the ``source/cfg/ntuples.json`` to have the correct directory for the
original signal ntuple files and the new files to save the modified trees on.
Run the script ``run_modify_trees.sh``. It should take several minutes.
This operation is only required once.

###### why are these operations needed
Current TTrees for signal have branches like:
- ``"C_Ds_pt"`` = [[1.2, 3.4],[2.3,1.7]]
- ``"C_pass_gen_matching"`` = [[1,0],[0,1]]
We only want those with ``C_pass_gen_matching`` == 1

The script modifies the ntuples to be
``"C_Ds_pt"`` = [[1.2],[1.7]]
##### categories
Run with the ``-- category`` argument to also add ``C_category`` to the trees.
There are six categories divided according to the charge of final state
muons and vertex significance, which are encoded in the variables:
- ``C_Hnl_vertex_2DSig_BS``
- ``C_mu_Hnl_charge``
-  ``C_mu_Ds_charge``

|| **Same sign** | **Opposite sign** |
|:------------------:|:-------------:|:-----------------:|
|    **vertex sig < 50**    |       1       |         4         |
| **50 < vertex sig < 150** |       2       |         5         |
|    **vertex sig > 150**   |       3       |         6         |


#### environments to run the scripts
Packages ROOT and tensorflow are required to run the ``run_TMVA.sh`` script.
The script ``run_mymva.sh`` requires tensorflow, scikit-learn, xgboost 
and usual python packages like numpy, pandas, matplotlib, etc.

### settings
#### variables
Modify the ``source/cfg/vars_new.json`` file to change the variables used
in the analysis. These will be used with the ``--new`` option in the scripts.
Modify the ``source/cfg/vars_old.json`` file to change the variables used
with the ``--old`` option in the scripts.
### Different methods
The methods you can run on are the ones listed in the 
``source/cfg/hyperparameters.json`` file.
- ``adabooost``: Boosted Decision Trees
- ``keras_shallow``: Shallow Neural Network with tensorflow.keras
- ``keras_deep``: Deep Neural Network with tensorflow.keras
- ``XGBoost``: XGBoost
#### hyperparameters
You can modify the hyperparameters for each method in the 
``source/cfg/hyperparameters.json`` file.
### Running the code (with categories)
Run the script ``source/training_my_mva_categories.py`` with the arguments:
- ``--mass`` to input the mass label like ``mN1p0`` (required)
- ``--out_dir`` to input the output directory (required)
- ``--train`` to train the models
- ``--results`` to produce the output results files
- ``--plots`` to produce the output plots

You can also directly run the script ``run_mymva_categories.sh`` where
the arguments are already set, and choose the list of masses to run on.
### Running the code (without categories) [OLD]
Run the script ``run_mymva.sh``
The arguments are:
- ``--train``: train the models
- ``--results``: produce results
- ``--new``: use new variables (either this or ``--old`` is required)
- ``--old``: use old variables (either this or ``--new`` is required)
- ``--out_dir``: output directory (required)

This runs a python implementation with some multi-variate analysis methods.
Some modifications can be done on the ``source/training_my_mva.py`` file e.g.
choose which methods to run.
####  TMVA code
Run the script ``run_TMVA.sh``

The arguments are:
- ``--mass``: mass label like mN1p0 (required)
- ``--new``: use new variables (either this or ``--old`` is required)
- ``--old``: use old variables (either this or ``--new`` is required)
- ``--transf``: perform D,G variables transformation in TMVA
- ``--out_dir``: output directory (required)

To run the TMVA code you must have run the normal MVA analysis first 
with the same (new or old) variables to have the keras models files 
that will be used in the TMVA code.

#### produce output plots
Run the script ``run_plots.sh``
The arguments are:
- ``--mymva``: produce plots for myMVA
- ``--tmva``: produce plots for TMVA
- ``--input_dir``: input directory with results of myMVA
- ``--input_dir_tmva``: input directory with results of TMVA
- ``--out_dir``: output directory for the plots (required)

