import pandas as pd
import json

df = pd.read_csv("../data/cleaned/final_dataset.csv")

output_file = "../data/cleaned/instructions.jsonl"

with open(output_file, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        instruction = f"Suggest a tourist place to visit in {row['country']}."
        input_text = f"Category: {row['category']}, Budget: {row['budget']}"
        output_text = (
            f"You can visit {row['place_name']}. "
            f"It is a {row['category']} attraction. "
            f"{row['description']}"
        )

        record = {
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        }

        f.write(json.dumps(record) + "\n")

print("Instruction data created ✅")
