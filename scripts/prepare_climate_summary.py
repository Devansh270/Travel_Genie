import pandas as pd

# load cleaned climate data
df = pd.read_csv("../data/cleaned/climate_cleaned.csv")

# convert date to datetime
df["dt"] = pd.to_datetime(df["dt"])

# extract month
df["month"] = df["dt"].dt.month

# average temperature by country & month
summary = (
    df.groupby(["Country", "month"])["AverageTemperature"]
      .mean()
      .reset_index()
)

# save summarized climate data
summary.to_csv("../data/cleaned/climate_summary.csv", index=False)

print("Climate summary created ✅")
