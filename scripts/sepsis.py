import os
import pandas as pd
import pm4py
import matplotlib.pyplot as plt

# create folders
os.makedirs("outputs/cleaned_data", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

print("Loading dataset...")

log = pm4py.read_xes("Sepsis Cases - Event Log.xes")
df = pm4py.convert_to_dataframe(log)

print("Dataset loaded")
print("Shape:", df.shape)

# stats before cleaning
num_events_before = len(df)
num_cases_before = df["case:concept:name"].nunique()
num_activities_before = df["concept:name"].nunique()

# fix column types
df["case:concept:name"] = df["case:concept:name"].astype(str)
df["concept:name"] = df["concept:name"].astype(str)
df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

# cleaning
invalid_timestamps = df["time:timestamp"].isna().sum()

df = df.dropna(subset=["case:concept:name", "concept:name", "time:timestamp"])

rows_before = len(df)
df = df.drop_duplicates(subset=["case:concept:name", "concept:name", "time:timestamp"])
duplicates_removed = rows_before - len(df)

rows_before = len(df)
df = df[df["concept:name"].str.strip() != ""]
empty_removed = rows_before - len(df)

df = df.sort_values(by=["case:concept:name", "time:timestamp"]).reset_index(drop=True)

# stats after cleaning
num_events_after = len(df)
num_cases_after = df["case:concept:name"].nunique()
num_activities_after = df["concept:name"].nunique()

trace_lengths = df.groupby("case:concept:name")["concept:name"].count()
avg_trace = round(trace_lengths.mean(), 2)

# save cleaned data
df.to_csv("outputs/cleaned_data/sepsis_cleaned.csv", index=False)

# summary table
summary = pd.DataFrame([
    {
        "stage": "before",
        "events": num_events_before,
        "cases": num_cases_before,
        "activities": num_activities_before,
        "avg_trace_length": avg_trace
    },
    {
        "stage": "after",
        "events": num_events_after,
        "cases": num_cases_after,
        "activities": num_activities_after,
        "avg_trace_length": avg_trace
    }
])

summary.to_csv("outputs/tables/sepsis_summary.csv", index=False)

# cleaning table
cleaning = pd.DataFrame([
    {
        "invalid_timestamps": invalid_timestamps,
        "duplicates_removed": duplicates_removed,
        "empty_activity_removed": empty_removed
    }
])

cleaning.to_csv("outputs/tables/sepsis_cleaning.csv", index=False)

# -----------------------------
# FIGURES (VERY IMPORTANT)
# -----------------------------

# 1. activity frequency
print("Plotting activity frequency...")
activity_counts = df["concept:name"].value_counts().head(20)

plt.figure()
activity_counts.plot(kind="bar")
plt.title("Top Activities - Sepsis")
plt.xlabel("Activity")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/figures/sepsis_activity_frequency.png")
plt.close()

# 2. trace length
print("Plotting trace length...")
trace_lengths = df.groupby("case:concept:name").size()

plt.figure()
trace_lengths.hist(bins=30)
plt.title("Trace Length Distribution - Sepsis")
plt.xlabel("Events per case")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/sepsis_trace_length.png")
plt.close()

# 3. case duration
print("Plotting case duration...")
case_times = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
case_times["duration_days"] = (
    case_times["max"] - case_times["min"]
).dt.total_seconds() / (60 * 60 * 24)

plt.figure()
case_times["duration_days"].hist(bins=30)
plt.title("Case Duration - Sepsis")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/sepsis_case_duration.png")
plt.close()

# 4. events over time
print("Plotting events over time...")
df["date"] = df["time:timestamp"].dt.date
events_per_day = df.groupby("date").size()

plt.figure()
events_per_day.plot()
plt.title("Events Over Time - Sepsis")
plt.xlabel("Date")
plt.ylabel("Events")
plt.tight_layout()
plt.savefig("outputs/figures/sepsis_events_over_time.png")
plt.close()

# final logs
print("Cleaning finished")
print("Events before:", num_events_before)
print("Events after:", num_events_after)
print("Duplicates removed:", duplicates_removed)
print("Invalid timestamps:", invalid_timestamps)
print("Sepsis analysis completed and saved.")