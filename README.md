# Process Mining Digital Twin Thesis

This repository contains the Python scripts used in the master thesis:

**Process Mining Approach for Building Process Digital Twins**

The scripts support:
- event-log preprocessing
- process discovery using Inductive Miner
- Petri-net model evaluation
- raw and cleaned event-log comparison
- final result-table generation

## Repository structure

scripts/bpi.py  
Cleaning and preprocessing of the BPI Challenge 2012 event log.

scripts/sepsis.py  
Cleaning and preprocessing of the Sepsis Cases event log.

scripts/insurrance1.py  
Cleaning and preprocessing of the Insurance Claims event log.

scripts/model_evaluation1.py  
Calculation of fitness, precision, F1-score, and fitting traces.

scripts/raw_vs_cleaned_experiment.py  
Comparison between raw and cleaned event-log configurations.

scripts/quality_impact_analysis1.py  
Generation of the quality impact analysis table.

scripts/final_analysis1.py  
Generation of the final model comparison table.

## Datasets

The original event logs are not included in this repository.  
They can be accessed from the dataset sources referenced in the thesis.

## Requirements

Python 3.x  
pandas  
pm4py  
matplotlib
