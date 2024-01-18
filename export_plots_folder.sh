#!/bin/bash


# Get the input directory path
input_dir="results_categories"

# Define the output directory path
output_dir="${input_dir}_exported_plots"

# Copy the directory structure while excluding files with "_model" or "_results" format
rsync -av --exclude '*_model.*' --exclude '*_results.*' --quiet "$input_dir/" "$output_dir/"
