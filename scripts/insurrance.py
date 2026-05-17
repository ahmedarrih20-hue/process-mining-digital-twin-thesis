import os
import pandas as pd
import matplotlib.pyplot as plt

# create folders
os.makedirs("outputs/cleaned_data", exist_ok=True)
os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv("Insurance_Event_Log.csv")

print("Dataset loaded")
print("Shape:", df.shape)

# rename columns
df = df.rename(columns={
    "Case ID": "case:concept:name",
    "Activity": "concept:name",
    "Timestamp": "time:timestamp"
})

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

# remove short cases
case_lengths = df.groupby("case:concept:name").size()
valid_cases = case_lengths[case_lengths >= 2].index
df = df[df["case:concept:name"].isin(valid_cases)]

# stats after cleaning
num_events_after = len(df)
num_cases_after = df["case:concept:name"].nunique()
num_activities_after = df["concept:name"].nunique()

trace_lengths = df.groupby("case:concept:name")["concept:name"].count()
avg_trace = round(trace_lengths.mean(), 2)

# save cleaned data
df.to_csv("outputs/cleaned_data/insurance_cleaned.csv", index=False)

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

summary.to_csv("outputs/tables/insurance_summary.csv", index=False)

# cleaning table
cleaning = pd.DataFrame([
    {
        "invalid_timestamps": invalid_timestamps,
        "duplicates_removed": duplicates_removed,
        "empty_activity_removed": empty_removed
    }
])

cleaning.to_csv("outputs/tables/insurance_cleaning.csv", index=False)

# 1. activity frequency
print("Plotting activity frequency...")
activity_counts = df["concept:name"].value_counts().head(20)

plt.figure()
activity_counts.plot(kind="bar")
plt.title("Top Activities - Insurance")
plt.xlabel("Activity")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("outputs/figures/insurance_activity_frequency.png")
plt.close()

# 2. trace length
print("Plotting trace length...")
trace_lengths = df.groupby("case:concept:name").size()

plt.figure()
trace_lengths.hist(bins=30)
plt.title("Trace Length Distribution - Insurance")
plt.xlabel("Events per case")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/insurance_trace_length.png")
plt.close()

# 3. case duration
print("Plotting case duration...")
case_times = df.groupby("case:concept:name")["time:timestamp"].agg(["min", "max"])
case_times["duration_days"] = (
    case_times["max"] - case_times["min"]
).dt.total_seconds() / (60 * 60 * 24)

plt.figure()
case_times["duration_days"].hist(bins=30)
plt.title("Case Duration - Insurance")
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("outputs/figures/insurance_case_duration.png")
plt.close()

# 4. events over time
print("Plotting events over time...")
df["date"] = df["time:timestamp"].dt.date
events_per_day = df.groupby("date").size()

plt.figure()
events_per_day.plot()
plt.title("Events Over Time - Insurance")
plt.xlabel("Date")
plt.ylabel("Events")
plt.tight_layout()
plt.savefig("outputs/figures/insurance_events_over_time.png")
plt.close()

print("Cleaning finished")
print("Events before:", num_events_before)
print("Events after:", num_events_after)
print("Duplicates removed:", duplicates_removed)
print("Invalid timestamps:", invalid_timestamps)
print("Insurance analysis completed and saved.")
