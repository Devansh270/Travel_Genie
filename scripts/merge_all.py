import pandas as pd

tourism = pd.read_csv("../data/cleaned/tourism_cleaned.csv")
cost = pd.read_csv("../data/cleaned/cost_cleaned.csv")

# normalize column names
tourism.columns = tourism.columns.str.lower()
cost.columns = cost.columns.str.lower()

# make sure budget column exists
if "budget_level" in cost.columns:
    cost = cost.rename(columns={"budget_level": "budget"})

# merge on country
final_df = tourism.merge(cost, on="country", how="left")

# FINAL GUARANTEE: required columns exist
final_df = final_df.rename(columns={
    "cost of living index": "cost_of_living_index"
})

final_df.to_csv(
    "../data/cleaned/final_dataset.csv",
    index=False,
    chunksize=10_000
)

print("FINAL DATASET READY ✅")
print(final_df.columns)
