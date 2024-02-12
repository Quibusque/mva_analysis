import pandas as pd
import numpy as np
import os

main_dir = '/home/quibus/mva_analysis/results_categories/myMVA'

#list directories in main_dir
dirs = [os.path.join(main_dir, d) for d in os.listdir(main_dir) if os.path.isdir(os.path.join(main_dir, d))]

for dir in dirs:
    #list dirs in dir
    catdirs = [os.path.join(dir, f"cat_{i}") for i in range(1, 7) if os.path.isdir(os.path.join(dir, f"cat_{i}"))]
    print(catdirs)
    break
