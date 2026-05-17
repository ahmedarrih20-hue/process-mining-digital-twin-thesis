import os
import pandas as pd

os.makedirs("outputs/tables", exist_ok=True)

df = pd.read_csv("outputs/tables/quality_impact_analysis.csv")

df["final_score"] = (
    (df["fitness"] * 0.4) +
    (df["precision"] * 0.4) +
    (df["f1_score"] * 0.2)
)

df["final_score"] = df["final_score"].round(4)

df = df.sort_values(by="final_score", ascending=False).reset_index(drop=True)
df["rank"] = range(1, len(df) + 1)

def final_conclusion(row):
    if row["rank"] == 1:
        return "Best overall model"
    elif row["fitness"] >= 0.95 and row["precision"] < 0.40:
        return "Good coverage but weak precision"
    elif row["precision"] >= 0.80:
        return "Strong and reliable model"
    else:
        return "Acceptable but can be improved"

df["final_conclusion"] = df.apply(final_conclusion, axis=1)

final_table = df[
    [
        "rank",
        "dataset",
        "events",
        "cases",
        "activities",
        "variants",
        "complexity_level",
        "fitness",
        "precision",
        "f1_score",
        "percentage_of_fitting_traces",
        "total_cleaning_changes",
        "final_score",
        "final_conclusion"
    ]
]
final_table.to_csv("outputs/tables/final_comparison.csv", index=False)

print("Final analysis finished")
print(final_table)
