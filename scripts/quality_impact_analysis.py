import os
import pandas as pd

os.makedirs("outputs/tables", exist_ok=True)

bpi_cleaning = pd.read_csv("outputs/tables/bpi_cleaning.csv")
sepsis_cleaning = pd.read_csv("outputs/tables/sepsis_cleaning.csv")
insurance_cleaning = pd.read_csv("outputs/tables/insurance_cleaning.csv")

discovery = pd.read_csv("outputs/tables/process_discovery_summary.csv")
evaluation = pd.read_csv("outputs/tables/model_evaluation_summary.csv")

cleaning = pd.DataFrame([
    {
        "dataset": "BPI",
        "invalid_timestamps": bpi_cleaning.loc[0, "invalid_timestamps"],
        "duplicates_removed": bpi_cleaning.loc[0, "duplicates_removed"],
        "empty_activity_removed": bpi_cleaning.loc[0, "empty_activity_removed"]
    },
    {
        "dataset": "Sepsis",
        "invalid_timestamps": sepsis_cleaning.loc[0, "invalid_timestamps"],
        "duplicates_removed": sepsis_cleaning.loc[0, "duplicates_removed"],
        "empty_activity_removed": sepsis_cleaning.loc[0, "empty_activity_removed"]
    },
    {
        "dataset": "Insurance",
        "invalid_timestamps": insurance_cleaning.loc[0, "invalid_timestamps"],
        "duplicates_removed": insurance_cleaning.loc[0, "duplicates_removed"],
        "empty_activity_removed": insurance_cleaning.loc[0, "empty_activity_removed"]
    }
])

final_table = discovery.merge(evaluation, on="dataset", how="inner")
final_table = final_table.merge(cleaning, on="dataset", how="left")

final_table["total_cleaning_changes"] = (
    final_table["invalid_timestamps"] +
    final_table["duplicates_removed"] +
    final_table["empty_activity_removed"]
)

def complexity_level(variants):
    if variants > 3000:
        return "High"
    elif variants > 900:
        return "Medium"
    else:
        return "Low"

def precision_level(value):
    if value >= 0.8:
        return "High"
    elif value >= 0.4:
        return "Medium"
    else:
        return "Low"

def fitness_level(value):
    if value >= 0.95:
        return "High"
    elif value >= 0.80:
        return "Medium"
    else:
        return "Low"

def short_interpretation(row):
    if row["fitness"] >= 0.95 and row["precision"] >= 0.80:
        return "Very good model quality"
    elif row["fitness"] >= 0.95 and row["precision"] < 0.40:
        return "High fitness but low precision"
    elif row["fitness"] >= 0.95:
        return "Good fitness with moderate precision"
    else:
        return "Model quality needs improvement"

final_table["complexity_level"] = final_table["variants"].apply(complexity_level)
final_table["fitness_level"] = final_table["fitness"].apply(fitness_level)
final_table["precision_level"] = final_table["precision"].apply(precision_level)
final_table["interpretation"] = final_table.apply(short_interpretation, axis=1)

final_table = final_table[
    [
        "dataset",
        "events",
        "cases",
        "activities",
        "variants",
        "complexity_level",
        "fitness",
        "fitness_level",
        "precision",
        "precision_level",
        "f1_score",
        "percentage_of_fitting_traces",
        "invalid_timestamps",
        "duplicates_removed",
        "empty_activity_removed",
        "total_cleaning_changes",
        "interpretation"
    ]
]

final_table.to_csv("outputs/tables/quality_impact_analysis.csv", index=False)

print("Quality impact analysis finished")
print(final_table)
