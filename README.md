# Process Mining Digital Twin Thesis

This repository contains the Python scripts and selected result files used in the master thesis:

**Process Mining Approach for Building Process Digital Twins**

The scripts support:
- event-log preprocessing
- process discovery using Inductive Miner
- Petri-net model evaluation
- raw and cleaned event-log comparison
- quality impact analysis
- result-table generation

## Repository structure

scripts/bpi.py  
Cleaning and preprocessing of the BPI Challenge 2012 event log.

scripts/sepsis.py  
Cleaning and preprocessing of the Sepsis Cases event log.

scripts/insurance.py  
Cleaning and preprocessing of the Insurance Claims event log.

scripts/discovery.py  
Process discovery using Inductive Miner and Petri-net export.

scripts/model_evaluation.py  
Calculation of fitness, precision, F1-score, and fitting-trace percentage.

scripts/raw_vs_cleaned_experiment.py  
Comparison between raw and cleaned event-log configurations.

scripts/quality_impact_analysis.py  
Generation of the quality impact analysis table by combining discovery, evaluation, and cleaning results.

## Results

Selected generated result files are stored in the results folder.

results/tables/  
Contains CSV tables with dataset summaries, cleaning results, model-quality evaluation results, and raw-versus-cleaned comparison results.

results/figures/  
Contains selected figures generated during the experiment.

## Datasets

The original event logs are not included in this repository.  
They can be accessed from the dataset sources referenced in the thesis.

## Requirements

Python 3.x  
pandas  
pm4py  
matplotlib  
numpy

## Notes

The scripts include comments explaining the main processing steps.

This repository is used as supporting material for the thesis and does not include the original datasets.
