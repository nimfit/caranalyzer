"""
clean_cars_data.py

Cleans cars.csv based on issues found during exploratory analysis:

- 9,145 exact duplicate rows
- Unrealistic/placeholder prices (< $500 or > $2,000,000)
- Unrealistic mileage values (> 400,000 miles)
- Zero mileage on pre-2020 vehicles likely represents missing data
- Pre-1980 vehicles with <1,000 miles are unusual but potentially legitimate,
  so they are flagged rather than removed
- mpg column stored as "city-highway" strings, with 0 used as a
  missing-value placeholder (ex. "0-0", "21-0")
- drivetrain has inconsistent spellings of the same category
- fuel_type has inconsistent spellings of the same category
- price_drop contains missing values; a separate flag identifies
  listings where a price drop is reported

Near-duplicate listings based on manufacturer/model/year/mileage are
intentionally kept because they may represent different sellers,
prices, or marketplace listings.
"""

import pandas as pd
import numpy as np


INPUT_PATH = "cars.csv"
OUTPUT_PATH = "cars_clean.csv"

# 1. Load data
# ----------------------------------------------------------------------

df = pd.read_csv(INPUT_PATH)

start_rows = len(df)

print(f"Loaded {start_rows:,} rows, {len(df.columns)} columns")


# 2. Drop exact duplicate rows
# ----------------------------------------------------------------------

before = len(df)

df = df.drop_duplicates()

print(f"Dropped {before - len(df):,} exact duplicate rows")

# Near-duplicates based on manufacturer/model/year/mileage are intentionally
# kept. These may represent separate listings from different sellers,
# different prices, or other legitimate marketplace records.



# 3. Standardize column types
# ----------------------------------------------------------------------

binary_cols = [
    "accidents_or_damage",
    "one_owner",
    "personal_use_only"
]

for col in binary_cols:
    # Nullable boolean preserves missing values as <NA>
    # while converting 0/1 to False/True.
    df[col] = df[col].astype("boolean")



# 4. Price cleaning
# ----------------------------------------------------------------------

# Listings below $500 are treated as likely data-entry or placeholder
# errors. This threshold may exclude a small number of genuinely
# inexpensive vehicles.
#
# Values above $2,000,000 are treated as extreme/outlier records.
# Genuine exotic vehicles can be expensive, so the threshold is
# intentionally high rather than using a typical used-car price cutoff.

before = len(df)

MIN_REASONABLE_PRICE = 500
MAX_REASONABLE_PRICE = 2_000_000

df = df[
    df["price"].between(
        MIN_REASONABLE_PRICE,
        MAX_REASONABLE_PRICE
    )
]

print(
    f"Dropped {before - len(df):,} rows with unrealistic price "
    f"(< ${MIN_REASONABLE_PRICE:,} or > ${MAX_REASONABLE_PRICE:,})"
)



# 5. Mileage cleaning
# ----------------------------------------------------------------------

# Mileage above 400,000 miles is treated as implausible for a vehicle
# currently being sold through a dealer/marketplace listing.
#
# The original dataset contains mileage values above 1,000,000 miles,
# including a maximum of approximately 1.12 million miles.

before = len(df)

MAX_REASONABLE_MILEAGE = 400_000

df = df[
    (df["mileage"].isna()) |
    (df["mileage"] <= MAX_REASONABLE_MILEAGE)
]

print(
    f"Dropped {before - len(df):,} rows with mileage > "
    f"{MAX_REASONABLE_MILEAGE:,}"
)



# 5a. Handle zero-mileage values
# ----------------------------------------------------------------------

# Zero mileage may be legitimate for new or nearly-new vehicles.
# For pre-2020 vehicles, however, zero mileage is more likely to
# represent an unreported/missing odometer value than a literal
# 0-mile reading.

old_zero_mileage = (
    (df["year"] <= 2019) &
    (df["mileage"] == 0)
)

zero_mileage_count = old_zero_mileage.sum()

df.loc[old_zero_mileage, "mileage"] = np.nan

print(
    f"Recoded {zero_mileage_count:,} zero-mileage readings on "
    f"pre-2020 vehicles as missing"
)



# 5b. Flag unusual classic-car mileage
# ----------------------------------------------------------------------

# Pre-1980 vehicles with fewer than 1,000 miles are unusual but
# potentially legitimate, such as garage-kept or collector vehicles.
# They are therefore retained and flagged instead of being removed.
#
# The original exploratory analysis identified 509 such records before
# cleaning. The final cleaned dataset contains the records that remain
# after the other cleaning steps.

df["flag_low_mileage_classic"] = (
    (df["year"] < 1980) &
    (df["mileage"] < 1000)
)

classic_flag_count = df["flag_low_mileage_classic"].sum()

print(
    f"Flagged {classic_flag_count:,} pre-1980 vehicles with "
    f"< 1,000 miles as flag_low_mileage_classic "
    f"(kept, not dropped)"
)



# 6. Split MPG into numeric city/highway columns
# ----------------------------------------------------------------------

# mpg is stored as "city-highway" strings.
# A value of 0 represents a missing value rather than 0 MPG.
#
# Example:
#   "39-38" -> city = 39, highway = 38
#   "21-0"  -> city = 21, highway = NaN
#   "0-0"   -> city = NaN, highway = NaN
#
# Some records contain unusual/reversed city and highway values.
# These are left unchanged rather than making an unsupported assumption.


def split_mpg(value):
    if pd.isna(value):
        return (np.nan, np.nan)

    parts = str(value).split("-")

    if len(parts) != 2:
        return (np.nan, np.nan)

    try:
        city, hwy = int(parts[0]), int(parts[1])
    except ValueError:
        return (np.nan, np.nan)

    city = np.nan if city == 0 else city
    hwy = np.nan if hwy == 0 else hwy

    return (city, hwy)


mpg_split = df["mpg"].apply(split_mpg)

df["mpg_city"] = mpg_split.apply(lambda x: x[0])
df["mpg_highway"] = mpg_split.apply(lambda x: x[1])

df = df.drop(columns=["mpg"])

print(
    "Split mpg into mpg_city / mpg_highway "
    "(0 recoded as missing)"
)



# 7. Standardize drivetrain categories
# ----------------------------------------------------------------------

drivetrain_map = {
    "fwd": "FWD",
    "front-wheel drive": "FWD",
    "front-wheel drive with limited-slip differential": "FWD",
    "front wheel drive": "FWD",

    "awd": "AWD",
    "all-wheel drive": "AWD",
    "all-wheel drive with locking and limited-slip differential": "AWD",
    "all-wheel drive with locking differential": "AWD",

    "4wd": "4WD",
    "four-wheel drive": "4WD",
    "four wheel drive": "4WD",
    "four-wheel drive with locking and limited-slip differential": "4WD",

    "rwd": "RWD",
    "rear-wheel drive": "RWD",
    "rear-wheel drive with limited-slip differential": "RWD",

    "unknown": np.nan,
}

original_drivetrain = df["drivetrain"]

df["drivetrain"] = (
    original_drivetrain
    .str.strip()
    .str.lower()
    .map(drivetrain_map)
    .fillna(original_drivetrain)
)



# 8. Standardize fuel_type categories
# ----------------------------------------------------------------------

# Only categories that can be confidently interpreted are standardized.
# Ambiguous values such as "B" are NOT automatically converted to Diesel
# because their meaning is not confirmed by the dataset documentation.

fuel_map = {
    "gasoline": "Gasoline",
    "gasoline fuel": "Gasoline",
    "regular unleaded": "Gasoline",
    "premium unleaded": "Gasoline",
    "g": "Gasoline",

    "diesel": "Diesel",
    "diesel fuel": "Diesel",

    "hybrid": "Hybrid",
    "hybrid fuel": "Hybrid",
    "gasoline/mild electric hybrid": "Hybrid",

    "plug-in hybrid": "Plug-In Hybrid",

    "electric": "Electric",

    "e85 flex fuel": "Flex Fuel",
    "flexible fuel": "Flex Fuel",

    "compressed natural gas": "CNG",

    "hydrogen fuel cell": "Hydrogen",

    "unspecified": np.nan,
    "other": np.nan,
    "gaseous": np.nan,
}

original_fuel_type = df["fuel_type"]

df["fuel_type"] = (
    original_fuel_type
    .str.strip()
    .str.lower()
    .map(fuel_map)
    .fillna(original_fuel_type)
)

# 9. Price drop
# ----------------------------------------------------------------------

# A separate indicator identifies whether a price-drop value is present.
#
# We do NOT automatically assume that a missing price_drop means that
# the vehicle never experienced a price drop. Missingness may instead
# represent unavailable information from the source.
#
# Therefore, price_drop remains NaN when no value is reported.

df["had_price_drop"] = df["price_drop"].notna()

print(
    "Added had_price_drop indicator; original missing price_drop "
    "values were retained as missing"
)


# 10. Save cleaned data
# ----------------------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print(
    f"\nFinal shape: {df.shape[0]:,} rows x {df.shape[1]:,} columns "
    f"(started at {start_rows:,} rows)"
)

print(f"Saved cleaned data to {OUTPUT_PATH}")
print("Original cars.csv was NOT modified.")

