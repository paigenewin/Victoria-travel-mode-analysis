import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# load raw datasets
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

household = pd.read_csv(DATA_DIR / "household_vista_2023_2024.csv")
person = pd.read_csv(DATA_DIR / "person_vista_2023_2024.csv")
journey_work = pd.read_csv(DATA_DIR / "journey_to_work_vista_2023_2024.csv")
journey_edu = pd.read_csv(DATA_DIR / "journey_to_education_vista_2023_2024.csv")

# box plot of household size to identify outliers 
household["hhsize"] = pd.to_numeric(household["hhsize"], errors="coerce")

hh_Q1 = household["hhsize"].quantile(0.25)
hh_Q3 = household["hhsize"].quantile(0.75)
hh_IQR = hh_Q3 - hh_Q1
hh_lower, hh_upper = hh_Q1 - 1.5 * hh_IQR, hh_Q3 + 1.5 * hh_IQR

hh_outliers = household[
    (household["hhsize"] < hh_lower) | 
    (household["hhsize"] > hh_upper)
]

print(f"Household size outliers detected: {len(hh_outliers)}")
plt.figure(figsize=(3,2))
sns.boxplot(x=household["hhsize"])
plt.title("Household Size Outliers")
plt.xlabel("Number of Household Members")
plt.show()


# detect invalid value / outlier of income level
print("Unique values for Household Income Level:")
print(household["hhinc_group"].dropna().unique())
print(f"\nTotal unique categories: {household['hhinc_group'].nunique()}\n")


# detect invalid value / outlier of age group
print("Unique values for Age Group:")
print(person["agegroup"].dropna().unique())
print(f"\nTotal unique categories: {person['agegroup'].nunique()}\n")

# investigate the values of journey_distance
negative_rows = journey_work[journey_work['journey_distance'] <= 0]
print (negative_rows)
print(f"\nTotal count of negative rows: {len(negative_rows)}")

# investigate the value of travle mode choice
other_count_edu = (journey_edu["main_journey_mode"] == "Other").sum()
print("\nCount of 'Other' in edu_df:", other_count_edu)
other_count_work = (journey_work["main_journey_mode"] == "Other").sum()
print("Count of 'Other' in work_df:", other_count_work)
