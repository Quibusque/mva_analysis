#!/bin/bash

# Define the log file
logfile="your_log_file.log"

# Use awk to keep only the last occurrence of lines starting with "Epoch"
awk '/^Epoch/{a[$1]=$0} END{for (i in a) print a[i]}' $logfile > temp.log

# Overwrite the old file with the sanitized version
mv temp.log $logfile