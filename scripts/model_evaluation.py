import os
import pandas as pd
import pm4py
# Create output folder
os.makedirs("outputs/tables", exist_ok=True)
results = []
# Define cleaned event logs and their corresponding discovered Petri-net models
datasets = [
    ("BPI", "outputs/cleaned_data/bpi_cleaned.csv", "outputs/models/bpi_model.pnml"),
    ("Sepsis", "outputs/cleaned_data/sepsis_cleaned.csv", "outputs/models/sepsis_model.pnml"),
    ("Insurance", "outputs/cleaned_data/insurance_cleaned.csv", "outputs/models/insurance_model.pnml")
]
for dataset_name, log_path, model_path in datasets:
    print(f"Evaluating {dataset_name}")
    
     # Load cleaned event log
    df = pd.read_csv(log_path)
    
    # Convert required columns to correct data types
    df["case:concept:name"] = df["case:concept:name"].astype(str)
    df["concept:name"] = df["concept:name"].astype(str)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")
    
    # Remove invalid rows and sort events by case and timestamp
    df = df.dropna(subset=["case:concept:name", "concept:name", "time:timestamp"])
    df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)
    
    # Format the dataframe as a PM4Py event log
    event_log = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp"
    )
    net, initial_marking, final_marking = pm4py.read_pnml(model_path)
    # Calculate fitness using token-based replay
    fitness_result = pm4py.fitness_token_based_replay(
        event_log,
        net,
        initial_marking,
        final_marking
    )
   # Calculate precision for the Petri-net model
    precision_value = pm4py.precision_token_based_replay(
        event_log,
        net,
        initial_marking,
        final_marking
    )
    # Calculate F1-score from fitness and precision
    if isinstance(fitness_result, dict):
        fitness_value = fitness_result.get("log_fitness", None)
        perc_fit_traces = fitness_result.get("percentage_of_fitting_traces", None)
    else:
        fitness_value = fitness_result
        perc_fit_traces = None

    if fitness_value is not None and precision_value is not None and (fitness_value + precision_value) > 0:
        f1_score = 2 * (fitness_value * precision_value) / (fitness_value + precision_value)
    else:
        f1_score = None
    # Store evaluation
    results.append({
        "dataset": dataset_name,
        "fitness": round(fitness_value, 4) if fitness_value is not None else None,
        "precision": round(precision_value, 4) if precision_value is not None else None,
        "f1_score": round(f1_score, 4) if f1_score is not None else None,
        "percentage_of_fitting_traces": round(perc_fit_traces, 2) if perc_fit_traces is not None else None
    })

    print(f"{dataset_name} done")
    print("Fitness:", round(fitness_value, 4) if fitness_value is not None else "None")
    print("Precision:", round(precision_value, 4) if precision_value is not None else "None")
# Save all model-quality
summary = pd.DataFrame(results)
summary.to_csv("outputs/tables/model_evaluation_summary.csv", index=False)

print("Evaluation finished")
print(summary)
