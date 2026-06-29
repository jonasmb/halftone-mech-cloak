#!/usr/bin/env bash
filename="../results/isodither.json"
python geometry_plot.py $filename
python mean_fluctuations.py $filename
python mean_fluctuations_plot.py $filename