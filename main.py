from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
import re

# =========================
# 1. MODEL SETUP
# =========================

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)
model.eval()

print("MODEL DEVICE:", next(model.parameters()).device)

# =========================
# 2. LOAD DATA
# =========================

df = pd.read_csv("data/cleaned/final_dataset_clean.csv")

df.columns = df.columns.str.strip().str.lower()

for col in ["place_name", "description", "category", "budget"]:
    df[col] = df[col].astype(str).str.lower()

# =========================
# 3. CHAT MEMORY
# =========================

current_itinerary = None
current_city = None
current_days = None

# =========================
# 4. USER QUERY PARSING
# =========================

def extract_days(query):
    m = re.search(r"(\d+)\s*day", query.lower())
    return int(m.group(1)) if m else 3


def extract_city(query):
    """
    Extract city from phrases like:
    - trip to mumbai
    - travel to new york
    - visit paris
    """
    query = query.lower()

    patterns = [
        r"to\s+([a-z\s]+)",
        r"visit\s+([a-z\s]+)",
        r"trip\s+to\s+([a-z\s]+)"
    ]

    for p in patterns:
        match = re.search(p, query)
        if match:
            return match.group(1).strip()

    return None


def extract_budget(query):
    q = query.lower()
    if "low" in q or "cheap" in q:
        return "low"
    if "high" in q or "luxury" in q:
        return "high"
    return "medium"


def extract_category(query):
    q = query.lower()
    if "historical" in q:
        return "historical"
    if "religious" in q:
        return "religious"
    if "nature" in q:
        return "nature"
    return "any"


def is_edit_query(query):
    return any(w in query.lower() for w in ["change", "edit", "modify", "update", "make"])

# =========================
# 5. ITINERARY GENERATION (SCALABLE CORE LOGIC)
# =========================

def generate_itinerary(city, days, budget, category):
    city = city.lower()

    def is_valid(row):
        text = (row["place_name"] + " " + row["description"]).lower()
        return city in text

    # 🔥 CORE FILTER (CITY VALIDATION)
    f = df[df.apply(is_valid, axis=1)]

    if f.empty:
        raise ValueError(f"No reliable places found for city: {city}")

    # Optional category filter
    if category != "any":
        f_cat = f[f["category"].str.contains(category, na=False)]
        if not f_cat.empty:
            f = f_cat

    # Optional budget filter
    f_budget = f[f["budget"].str.contains(budget, na=False)]
    if not f_budget.empty:
        f = f_budget

    return f.sample(min(days, len(f)))


def format_itinerary(itin):
    return "\n".join(
        f"Day {i}: Visit {r.place_name.title()}. {r.description}"
        for i, r in enumerate(itin.itertuples(), 1)
    )

# =========================
# 6. LLM CALL
# =========================

def call_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.4,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# =========================
# 7. MAIN CHAT LOOP
# =========================

print("\n🌍 Travel Genie (type 'exit' to quit)\n")

while True:
    user_query = input("You: ")
    if user_query.lower() == "exit":
        break

    # ---- EDIT MODE ----
    if current_itinerary and is_edit_query(user_query):
        prompt = f"""
You are editing an EXISTING itinerary.

STRICT RULES:
- City MUST remain {current_city}
- Total days MUST remain {current_days}
- Do NOT add new cities

CURRENT ITINERARY:
{current_itinerary}

USER REQUEST:
{user_query}
"""
        updated = call_llm(prompt)
        current_itinerary = updated
        print("\nAssistant:\n", updated)
        continue

    # ---- NEW PLAN ----
    city = extract_city(user_query)
    if city is None:
        print("\n❌ Please specify a city (e.g. trip to Mumbai)\n")
        continue

    days = extract_days(user_query)
    budget = extract_budget(user_query)
    category = extract_category(user_query)

    try:
        itin_df = generate_itinerary(city, days, budget, category)
    except Exception as e:
        print(f"\n❌ {e}\n")
        continue

    base_itin = format_itinerary(itin_df)

    prompt = f"""
You are a travel itinerary generator.

STRICT RULES:
- City MUST be {city}
- Do NOT mention any other city

Trip Details:
City: {city}
Days: {days}
Budget: {budget}
Category: {category}

Use ONLY the places below and enhance them:

{base_itin}
"""

    response = call_llm(prompt)

    current_itinerary = response
    current_city = city
    current_days = days

    print("\nAssistant:\n", response)
