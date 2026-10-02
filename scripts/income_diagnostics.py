import pandas as pd  
from pathlib import Path 

hh_income_missing_values = {"", "NA", "N/A", "Unknown", "Prefer not to say", "Not stated"}

# tokens for employment and income flags
nlf_tokens = ["not in labour", "not in labor", "retired", "home duties", "student - not working", "nlf", "not in work force", "full time tafe", "uni", "primary school", "not yet at school", "keeping house", "secondary school"]
unemployed_tokens = ["unemployed"]
person_no_income_tokens = ["nil income", "$0", "0", "zero", "negative income"]

# load raw datasets 
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
hh = pd.read_csv(DATA_DIR / "household_vista_2023_2024.csv")
pp = pd.read_csv(DATA_DIR / "person_vista_2023_2024.csv")

# flag household with missing income 

def is_missing_household_income(val):
    if pd.isna(val):
        return True
    s = str(val).strip().lower()
    return s in {v.lower() for v in hh_income_missing_values}

hh["_hh_income_missing"] = hh["hhinc_group"].apply(is_missing_household_income)


# flag person with no income
def has_no_personal_income(v):
    if pd.isna(v):
        # if missing personal income likely means "unknown", treat as False (not confirmed no income)
        return False
    s = str(v).strip().lower()
    if s in {"0", "0.0", "$0", "$0.00"}:
        return True
    return any(tok in s for tok in [t.lower() for t in person_no_income_tokens])

def is_nlf_or_unemployed(v):
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    if any(tok in s for tok in [t.lower() for t in unemployed_tokens]):
        return True
    if any(tok in s for tok in [t.lower() for t in nlf_tokens]):
        return True
    if s in {"not in labour force", "not in labor force", "unemployed", "not in work force"}:
        return True
    return False

pp["_no_person_income_flag"] = pp["persinc"].apply(has_no_personal_income) 
pp["_nlf_or_unemp_flag"]     = pp["emptype"].apply(is_nlf_or_unemployed)
pp["_meets_condition"]       = pp["_no_person_income_flag"] & pp["_nlf_or_unemp_flag"]


# aggregate to household level
grp = pp.groupby("hhid").agg(
    members_total=("persid", "count"),
    members_meet=("_meets_condition", "sum")
)
grp["all_members_meet"] = (grp["members_total"] > 0) & (grp["members_meet"] == grp["members_total"])
grp["some_members_meet"] = (grp["members_meet"] > 0) & (grp["members_meet"] < grp["members_total"])


# merge
merged = hh.merge(grp, how="left", on="hhid")

# Keep only households with missing income
missing_income_hh = merged[merged["_hh_income_missing"] == True].copy()

# Diagnose reason categories
def reason(row):
    if pd.isna(row["members_total"]):
        return "No person records found"
    if row["all_members_meet"]:
        return "All members have no income & not working"
    elif row["some_members_meet"]:
        return "Some members no income & not working"
    else:
        return "At least one member is working"

missing_income_hh["missing_reason"] = missing_income_hh.apply(reason, axis=1)


# summarise results 

summary = (
    missing_income_hh["missing_reason"]
    .value_counts()
    .rename_axis("Reason")
    .reset_index(name="Number of households")
)
print("\nSummary of Missing Household Income Reasons")
print(summary)
print("\n")

# Save detailed CSV
cols = [
    "hhid", "hhinc_group", "members_total", "members_meet", "all_members_meet", "some_members_meet", "missing_reason"
]

detailed_preview = missing_income_hh[cols].sort_values("hhid").head(20)
print("Detailed preview (first 20 households):")
print(detailed_preview.to_string(index=False))
