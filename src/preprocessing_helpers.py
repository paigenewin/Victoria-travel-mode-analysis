import pandas as pd
import numpy as np
import re

def mode_group(journey_df, mode_col="main_journey_mode"):
    """
    Group individual travel modes into three categories: Public, Private, and Active.
    """

    # Make a copy of the input DataFrame to avoid modifying the original data
    df = journey_df.copy()
    new_col = "mode_group"

    # Convert the travel mode column to string and remove extra whitespace
    mode_series = df[mode_col].astype(str).str.strip()

    # Categorize each specific travel options in to mian modes
    # Public transport modes
    public_modes = {"Train", "Tram", "Public Bus", "School Bus"}


    # Private modes
    private_modes = {"Vehicle Driver", "Vehicle Passenger", "Motorcycle", "Taxi / Rideshare"}


    # Active modes
    active_modes = {"Walking", "Bicycle", "Running/jogging", "e-Scooter", "Mobility Scooter", "Scooter"}


    # Map each travel mode to its corresponding group
    def to_group(label):
        if label in public_modes:
            return "Public"
        if label in private_modes:
            return "Private"
        if label in active_modes:
            return "Active"
        return np.nan # Assign NaN if the mode does not belong to any group

    # Apply the mapping function to create the new 'mode_group' column
    df[new_col] = mode_series.apply(to_group)
    # Convert the new column to categorical type with a defined order
    df[new_col] = pd.Categorical(
        df["mode_group"],
        categories=["Public", "Private", "Active"],
        ordered=False
    )
    return df



def bike_ownership(household_df, bikes_col="totalbikes", size_col="hhsize"):
    """
    Categorize households based on bike ownership level relative to household size.
    - totalbikes >= hhsize → Sufficient
    - totalbikes < hhsize → Insufficient
    """

    # Create a copy of the input DataFrame to avoid modifying the original data
    household_df = household_df.copy()
    new_col = "bike_ownership"

    # Convert the 'totalbikes' and 'hhsize' columns to numeric types
    # Invalid or missing values are coerced to NaN, then filled with 0 for 'totalbikes'
    household_df[bikes_col] = pd.to_numeric(household_df[bikes_col], errors="coerce").fillna(0)
    household_df[size_col] = pd.to_numeric(household_df[size_col], errors="coerce")

    # Define a function to classify each household's bike ownership category
    def bike_group(bike, size):
        if bike == 0:
            return "No bikes"
        if bike > 0 and bike < size:
            return "Insufficient bikes"
        if bike >= size:
            return "Sufficient bikes"
        return np.nan  # Return NaN if data is invalid or missing

    # Apply the classification function
    household_df[new_col] = household_df[[bikes_col, size_col]].apply(
        lambda r: bike_group(r[bikes_col], r[size_col]), axis=1
    )

    # Convert the new column to categorical type with defined order
    household_df[new_col] = pd.Categorical(
        household_df[new_col],
        categories=["No bikes", "Insufficient bikes", "Sufficient bikes"],
        ordered=False
    )

    # Return the updated DataFrame with the new 'bike_ownership' column
    return household_df




def clean_income_level(household_df, income_col = "hhinc_group"):
    """
    Clean the income group:
    - Convert every missing values to NaN
    - Create a new column 'income level' that keeps orginal ranges but replace missing values as Unknown
    """

    # Create a copy of the input DataFrame to avoid modifying the original data
    household_df = household_df.copy()
    new_col = "income_level"

    # Define a set of values representing missing or invalid income responses
    missing_value = {"", "NA", "N/A", "Refused", "Prefer not to say", "Unknown", "nan"}

    # Convert the income column to string and remove extra spaces
    household_df[income_col] = household_df[income_col].astype(str).str.strip()
    # replace missing values by NaN
    household_df[income_col] = household_df[income_col].replace(list(missing_value), np.nan)
    household_df[new_col] = household_df[income_col].fillna(0)
    return household_df


    
    
def five_yrs_to_decade(age):
    """
    Convert an age group from 5-year intervals into decade-based bins.

    Example:
    "25–29" → "20->29"
    "40-44" → "40->49"
    """

    # If the input value is missing, return it unchanged
    if pd.isna(age):
      return age

    # Extract numeric values from the input string
    nums = re.findall(r"\d+", str(age))

    # If no numeric values are found, return the original input
    if not nums:
        return age

    # convert from 5 years to decade-binned
    start = int(nums[0])
    decade_start = (start // 10) * 10
    decade_end = decade_start + 9

    # Return the new age range in "start->end" format
    return f"{decade_start}->{decade_end}"


    
def age_bin(person_df, age_col = "agegroup"):
    """
    Categorize age groups into decade-based bins.

    Example:
    "25–29" → "20->29"
    "80–84" or higher → "70+"
    """

    # Create a copy of the input DataFrame to avoid modifying the original data
    df = person_df.copy()
    new_col = "age_binned"

    # Apply the five_yrs_to_decade function to convert 5-year intervals into decade bins
    df[new_col] = df[age_col].apply(five_yrs_to_decade)
    # Combine all ages 70 and above into a single "70+" category
    df[new_col] = df[new_col].replace(["70->79", "80->89", "90->99", "100->109"], "70+")
    # Define the order of age categories from youngest to oldest
    ordered = [f"{d*10}->{d*10+9}" for d in range(0,7)] + ["70+"]
    # Convert the new column to categorical type with defined order
    df[new_col] = pd.Categorical(df[new_col], categories = ordered, ordered=True)
    return df

def emp_status_bin(persons_df, mainact_column="mainact"):
    """
    categorise 'mainact' into 5 groups:
    { 'Full time', 'Part time', 'Student', 'Not in labour force/Retired', 'Unemployment' }
    """

    # Create a copy of the input DataFrame to avoid modifying the original data
    persons_df = persons_df.copy()
    new_col = "employment_binned"

    mapping = {
        # Work
        "Full-time Work": "Full time",
        "Part-time Work": "Part time",
        "Casual Work": "Part time",

        # Students
        "Primary School": "Student",
        "Secondary School": "Student",
        "Full-time TAFE/Uni": "Student",
        "Part-time TAFE/Uni": "Student",
        "Other Education": "Student",

        # Not in labour force / Retired
        "Not Yet at School": "Not in labour force/Retired",
        "Keeping House": "Not in labour force/Retired",
        "Retired": "Not in labour force/Retired",
        "Volunteer": "Not in labour force/Retired",
        "Other": "Not in labour force/Retired",

        # Unemployment
        "Unemployed": "Unemployment"
    }

    # Map original activity values to new grouped employment categories
    persons_df[new_col] = (
        persons_df[mainact_column]
        .map(mapping)
        .fillna(np.nan)  # keep unknown or unexpected values as NaN
    )

    # Convert the new column to categorical type with defined group order
    persons_df[new_col] = pd.Categorical(
        persons_df[new_col],
        categories=[
            "Full time",
            "Part time",
            "Student",
            "Not in labour force/Retired",
            "Unemployment"
        ],
        ordered=False
    )

    return persons_df

def private_veh_ownership(household_df,
                  person_df,
                  household_id_col = "hhid",
                  vehicles_col = "totalvehs",
                  person_household_id_col = "hhid",
                  has_car_licence_col = "carlicence"):
    """
    Determine private vehicle ownership at the household level
    Strict ownership:
    - totalvehs >= 1
    - any household members has licence
    """

    # Create a copy of the input DataFrame to avoid modifying the original data
    household_df = household_df.copy()
    person_copy = person_df.copy()

    # Convert vehicle count to numeric and mark households that own >= 1 vehicle
    household_df[vehicles_col] = pd.to_numeric(household_df[vehicles_col], errors = "coerce")
    owns_vehs_series = household_df[vehicles_col].fillna(0) >= 1

    # Map detailed licence status to a Yes/No
    mapping = {
        "Full Licence": "Yes",
        "Green Probationary Licence": "Yes",
        "Red Probationary Licence": "No",
        "Learner Permit": "No",
        "No Car Licence": "No",
    }

    person_copy[has_car_licence_col] = (
    person_copy[has_car_licence_col]
      .map(mapping)
      .fillna(np.nan)  # keep unknown or unexpected values as NaN
    )

    # Convert the Yes/No licence flag to boolean
    person_copy[has_car_licence_col] = (
        person_copy[has_car_licence_col].eq("Yes")
    )

    # Find if there is at least 1 member has driver licence per household
    any_licensed_by_household = (
        person_copy.groupby(person_household_id_col, dropna = False)[has_car_licence_col].any().rename("any_licensed_member")
    )


    # Join the per-household licence flag back to the household table
    household_df = household_df.merge(
        any_licensed_by_household,
        how = "left",
        left_on = household_id_col,
        right_index = True
    )

    # Compute strict ownership: owns >=1 vehicle AND any member licensed
    # If a household has no person records, treat as no licensed member (fillna(False))
    household_df["car_ownership"] = owns_vehs_series & household_df["any_licensed_member"].fillna(False)
    return household_df

def household_size_bin (household_df, size_col = "hhsize"):
    """
    categorise household size
    """
    # Create a copy of the input DataFrame to avoid modifying the original data
    household_df = household_df.copy()
    new_col = "hhsize_binned"

    # Convert the household size column to numeric type
    household_df[size_col] = pd.to_numeric(household_df[size_col], errors = "coerce")

    # Define a function to map household size into categories
    def size_to_bin(size_val):
        if pd.isna(size_val):
            return np.nan
        if size_val <= 1:
            return "1"
        if size_val == 2:
            return "2"
        if 3 <= size_val <= 4:
            return "3-4"
        return "5+"

    # Apply the mapping function to create the new binned household size column
    household_df[new_col] = household_df[size_col].apply(size_to_bin)
    # Convert the new column to categorical type with defined order
    household_df[new_col] = pd.Categorical(
        household_df[new_col],
        categories = ["1", "2", "3-4", "5+"],
    )
    return household_df

def region_bin(household_df, region_col = 'homesubregion_ASGS'):
    """
    Categorize household regions into four spatial bins based on subregion names
    """

    # Create a copy of the input DataFrame to avoid modifying the original data
    household_df = household_df.copy()
    new_col = "region_binned"

    # # Convert region names to lowercase for consistent string matching
    lower_region = household_df[region_col].astype(str).str.lower()

    # Define a function to assign region categories based on keywords
    def region_to_bin(val):
        if "inner" in val:
            return "Inner"
        if 'middle' in val:
            return "Middle"
        if 'outer' in val:
            return "Outer"
        if 'other' in val:
            return "Other"
        return np.nan

    # Apply the mapping function to create a new region category column
    household_df[new_col] = lower_region.apply(region_to_bin)

    # Convert the new column to categorical type with a defined order
    household_df[new_col] = pd.Categorical (
        household_df[new_col],
        categories = ["Inner", "Middle", "Outer", "Other"],
        ordered = True
    )
    return household_df

def distance_bin(df, distance_col='journey_distance'):
    """
    Categorize journey distance into fixed distance ranges (in kilometers).
    """
    bins = [0, 50, 100, 150, 200, 250, 300]
    labels = ["0–50", "50–100", "100–150", "150–200", "200–250", "250+"]
    # Use pandas.cut() to assign each record to the appropriate distance range
    df["distance_group"] = pd.cut(df["journey_distance"], bins=bins, labels=labels, right=False)
    return df
