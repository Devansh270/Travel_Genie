from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pandas as pd
import re
import os
from openai import OpenAI


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def chatgpt_fallback(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content


model_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
model.eval()


final_df = pd.read_csv("data/cleaned/final_dataset.csv")
climate = pd.read_csv("data/cleaned/climate_summary.csv")


def extract_days(q):
    m = re.search(r"(\d+)\s*[-]?\s*day[s]?", q)
    return int(m.group(1)) if m else 2


def parse_user_query(query):
    q = query.lower()

    country = "India"
    if "usa" in q or "united states" in q:
        country = "United States"
    elif "iran" in q:
        country = "Iran"

    if "historical" in q:
        category = "Historical"
    elif "religious" in q:
        category = "Religious"
    elif "nature" in q:
        category = "Nature"
    else:
        category = "Any"

    if "low" in q:
        budget = "Low"
    elif "high" in q:
        budget = "High"
    else:
        budget = "Medium"

    days = extract_days(q)
    return country, category, budget, days


def has_data(df, country, category, budget):
    if category == "Any":
        f = df[(df.country == country) & (df.budget == budget)]
    else:
        f = df[(df.country == country) & (df.category == category) & (df.budget == budget)]
    return not f.empty


def generate_itinerary(df, country, category, budget, days):
    if category == "Any":
        f = df[(df.country == country) & (df.budget == budget)]
    else:
        f = df[(df.country == country) & (df.category == category) & (df.budget == budget)]

    if f.empty:
        f = df[df.country == country]

    return f.sample(min(days, len(f)))

def format_itinerary(itin):
    return "\n".join(
        f"Day {i}: Visit {r.place_name}. {r.description}"
        for i, r in enumerate(itin.itertuples(), 1)
    )


def climate_summary(country):
    d = climate[climate.Country == country]
    return (
        f"Climate: Temperature ranges between "
        f"{d.AverageTemperature.min():.1f}°C and {d.AverageTemperature.max():.1f}°C."
    )


def hallucinated(answer, places):
    a = answer.lower()
    return not any(p.lower() in a for p in places)


query = input("Enter your travel query: ")

country, category, budget, days = parse_user_query(query)
itin_df = generate_itinerary(final_df, country, category, budget, days)
itin_text = format_itinerary(itin_df)

prompt = f"""
Create a {days}-day travel itinerary for {country}.
Category: {category}, Budget: {budget}

{itin_text}
"""

if has_data(final_df, country, category, budget):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=150)

    answer = tokenizer.decode(out[0], skip_special_tokens=True)
    places = itin_df.place_name.tolist()

    if hallucinated(answer, places):
        answer = chatgpt_fallback(prompt)
        source = "CHATGPT (hallucination fallback)"
    else:
        source = "MISTRAL"
else:
    answer = chatgpt_fallback(prompt)
    source = "CHATGPT (no data fallback)"

final_answer = answer + "\n\n" + climate_summary(country)

print(f"\n[SOURCE: {source}]\n")
print(final_answer)
