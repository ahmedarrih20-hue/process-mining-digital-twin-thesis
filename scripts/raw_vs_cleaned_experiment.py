import os
import pandas as pd
import pm4py

os.makedirs("outputs/models/raw_vs_cleaned", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)

def prepare_log(df, minimal=True):

    df = df.copy()
 # Convert required event-log columns to the correct types
    df["case:concept:name"] = df["case:concept:name"].astype(str)
    df["concept:name"] = df["concept:name"].astype(str)
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

    # Essential formatting: PM4Py cannot work correctly without valid case, activity, timestamp
    df = df.dropna(subset=["case:concept:name", "concept:name", "time:timestamp"])

    if not minimal:
        # Cleaning operations
        df = df.drop_duplicates(
            subset=["case:concept:name", "concept:name", "time:timestamp"]
        )
        df = df[df["concept:name"].str.strip() != ""]

        # Remove short cases
        case_lengths = df.groupby("case:concept:name").size()
        valid_cases = case_lengths[case_lengths >= 2].index
        df = df[df["case:concept:name"].isin(valid_cases)]

    df = df.sort_values(
        by=["case:concept:name", "time:timestamp"]
    ).reset_index(drop=True)

    return df


def evaluate_dataset(dataset_name, version_name, df):
    """
    Discover model and evaluate it using fitness, precision, and F1-score.
    """

    print(f"Running {dataset_name} - {version_name}")

    event_log = pm4py.format_dataframe(
        df,
        case_id="case:concept:name",
        activity_key="concept:name",
        timestamp_key="time:timestamp"
    )

    # Process discovery into a Petri net
    process_tree = pm4py.discover_process_tree_inductive(event_log)
    net, initial_marking, final_marking = pm4py.convert_to_petri_net(process_tree)

    # Save PNML model
    model_path = f"outputs/models/raw_vs_cleaned/{dataset_name.lower()}_{version_name.lower()}_model.pnml"
    pm4py.write_pnml(net, initial_marking, final_marking, model_path)

    # Save Petri-net visualization
    image_path = f"outputs/models/raw_vs_cleaned/{dataset_name.lower()}_{version_name.lower()}_model.png"
    pm4py.save_vis_petri_net(net, initial_marking, final_marking, image_path)

    # Calculate fitness using token-based replay
    fitness_result = pm4py.fitness_token_based_replay(
        event_log,
        net,
        initial_marking,
        final_marking
    )

    precision_value = pm4py.precision_token_based_replay(
        event_log,
        net,
        initial_marking,
        final_marking
    )

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

    variants = df.groupby("case:concept:name")["concept:name"].apply(tuple).nunique()

    return {
        "dataset": dataset_name,
        "version": version_name,
        "events": len(df),
        "cases": df["case:concept:name"].nunique(),
        "activities": df["concept:name"].nunique(),
        "variants": variants,
        "fitness": round(fitness_value, 4) if fitness_value is not None else None,
        "precision": round(precision_value, 4) if precision_value is not None else None,
        "f1_score": round(f1_score, 4) if f1_score is not None else None,
        "percentage_of_fitting_traces": round(perc_fit_traces, 2) if perc_fit_traces is not None else None,
        "model_file": model_path,
        "image_file": image_path
    }


# Load raw datasets

# BPI raw
bpi_log = pm4py.read_xes("BPI_Challenge_2012.xes")
bpi_raw = pm4py.convert_to_dataframe(bpi_log)

# Sepsis raw
sepsis_log = pm4py.read_xes("Sepsis Cases - Event Log.xes")
sepsis_raw = pm4py.convert_to_dataframe(sepsis_log)

# Insurance raw
insurance_raw = pd.read_csv("Insurance_Event_Log.csv")
insurance_raw = insurance_raw.rename(columns={
    "Case ID": "case:concept:name",
    "Activity": "concept:name",
    "Timestamp": "time:timestamp"
})

# Load cleaned datasets and Prepare logs
bpi_cleaned = pd.read_csv("outputs/cleaned_data/bpi_cleaned.csv")
sepsis_cleaned = pd.read_csv("outputs/cleaned_data/sepsis_cleaned.csv")
insurance_cleaned = pd.read_csv("outputs/cleaned_data/insurance_cleaned.csv")

# Define raw and cleaned configurations for each dataset
datasets = [                                      
    ("BPI", "Raw", prepare_log(bpi_raw, minimal=True)),
    ("BPI", "Cleaned", prepare_log(bpi_cleaned, minimal=False)),

    ("Sepsis", "Raw", prepare_log(sepsis_raw, minimal=True)),
    ("Sepsis", "Cleaned", prepare_log(sepsis_cleaned, minimal=False)),

    ("Insurance", "Raw", prepare_log(insurance_raw, minimal=True)),
    ("Insurance", "Cleaned", prepare_log(insurance_cleaned, minimal=False)),
]

results = []
# Run discovery and evaluation
for dataset_name, version_name, df in datasets:
    result = evaluate_dataset(dataset_name, version_name, df)
    results.append(result)
# Save raw-versus-cleaned comparison results
comparison = pd.DataFrame(results)
comparison.to_csv(
    "outputs/tables/raw_vs_cleaned_model_quality.csv",
    index=False
)

print("Raw vs cleaned experiment finished")
print(comparison)
