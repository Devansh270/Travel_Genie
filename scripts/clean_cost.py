import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

ALLOWED = ["India", "United States", "Iran"]

raw_path = BASE_DIR / "data" / "raw" / "raw_cost.csv"
clean_path = BASE_DIR / "data" / "cleaned" / "cost_cleaned.csv"

df = pd.read_csv(raw_path)

# normalize column names
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace(" ", "_")
)

df = df[df["country"].isin(ALLOWED)]
df = df.dropna()

def budget(cost):
    if cost < 40:
        return "Low"
    elif cost < 70:
        return "Medium"
    else:
        return "High"

# ✅ correct column
df["budget_level"] = df["cost_index"].apply(budget)

df.to_csv(clean_path, index=False)

print("Cost cleaned successfully ✅")
