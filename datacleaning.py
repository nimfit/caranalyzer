"""
datacleaning.py

Reviews the dataset in an exploratory analysis that shows duplicates, extreme values, etc.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


print("Loading dataset: ")
df = pd.read_csv("cars.csv")
print("Dataset loaded.")

print(f"Number of rows: {len(df)}")
print(f"Number of rows: {len(df.columns)}")

print("\nColumns:")

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

print("\nData types:")
print(df.dtypes)

print("Missing Values")
print("_____________________________________________")

missing = df.isna().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_pct
}).sort_values("missing_count", ascending=False)

print("\nMissing values by column:")
print(missing_report)

print("\nColumns with >25% missing:")
print(
    missing_report[missing_report["missing_percent"] > 25]
)

print("\nColumns with >50% missing:")
print(
    missing_report[missing_report["missing_percent"] > 50]
)

print("Exact Duplicates")
print("_____________________________________________")

duplicate_ct = df.duplicated().sum()

print(f"\nExact duplicate rows: {duplicate_ct:,}")
print(f"Percentage duplicated: {duplicate_ct / len(df) * 100:.2f}%")

if duplicate_ct > 0:
    print("\nExample duplicate rows:")

    duplicate_examples = df[
        df.duplicated(keep=False)
    ].sort_values(
        list(df.columns)
    ).head(20)

    print(duplicate_examples)


print("\n Potential Duplicate Vehicles")
print("_____________________________________________")

print(
    "Duplicate manufacturer/model/year/mileage combinations:",
    df.duplicated(
        subset=["manufacturer", "model", "year", "mileage"],
        keep=False
    ).sum()
)

print(
    df[
        df.duplicated(
            subset=["manufacturer", "model", "year", "mileage"],
            keep=False
        )
    ][
        ["manufacturer", "model", "year", "mileage", "price"]
    ].head(50)
)



print("\nPrice Analysis")
print("_____________________________________________")
print(df["price"].describe())

print(
    df["price"].quantile(
        [0, .001, .005, .01, .05, .25, .50, .75, .95, .99, .995, .999, 1]
    )
)

print("Price <= 0:", (df["price"] <= 0).sum())
print("Price < $1,000:", (df["price"] < 1000).sum())
print("Price < $2,000:", (df["price"] < 2000).sum())
print("Price > $100,000:", (df["price"] > 100000).sum())
print("Price > $200,000:", (df["price"] > 200000).sum())
print("Price > $250,000:", (df["price"] > 250000).sum())
print("Price > $500,000:", (df["price"] > 500000).sum())
print("Price > $1,000,000:", (df["price"] > 1000000).sum())



print("\nLowest-Priced Vehicles")
print("_____________________________________________")

print(
    df.nsmallest(
        50,
        "price"
    )[
        ["manufacturer", "model", "year", "mileage", "price"]
    ]
)



print("\nHighest-Priced Vehicles")
print("_____________________________________________")

print(
    df.nlargest(
        50,
        "price"
    )[
        ["manufacturer", "model", "year", "mileage", "price"]
    ]
)



print("\nMileage Analysis")
print("_____________________________________________")

print(df["mileage"].describe())

print(
    df["mileage"].quantile(
        [0, .001, .005, .01, .05, .25, .50, .75, .95, .99, .995, .999, 1]
    )
)

print("Mileage missing:", df["mileage"].isna().sum())
print("Mileage = 0:", (df["mileage"] == 0).sum())
print("Mileage < 0:", (df["mileage"] < 0).sum())
print("Mileage > 100,000:", (df["mileage"] > 100000).sum())
print("Mileage > 200,000:", (df["mileage"] > 200000).sum())
print("Mileage > 300,000:", (df["mileage"] > 300000).sum())
print("Mileage > 500,000:", (df["mileage"] > 500000).sum())
print("Mileage > 1,000,000:", (df["mileage"] > 1000000).sum())



print("\nHighest-Mileage Vehicles")
print("_____________________________________________")

print(
    df.nlargest(
        50,
        "mileage"
    )[
        ["manufacturer", "model", "year", "mileage", "price"]
    ]
)



print("\n--- Year Analysis ---")
print(df["year"].describe())

print("Year < 1980:", (df["year"] < 1980).sum())
print("Year < 1990:", (df["year"] < 1990).sum())
print("Year < 2000:", (df["year"] < 2000).sum())
print("Year > 2024:", (df["year"] > 2024).sum())
print("Year > 2026:", (df["year"] > 2026).sum())

print("\nOldest vehicles:")
print(
    df.nsmallest(
        30,
        "year"
    )[
        ["manufacturer", "model", "year", "mileage", "price"]
    ]
)



print("\nSuspicious Year + Mileage")
print("_____________________________________________")


print(
    "2020+ vehicles with >300k miles:",
    (
        (df["year"] >= 2020) &
        (df["mileage"] > 300000)
    ).sum()
)

print(
    df[
        (df["year"] >= 2020) &
        (df["mileage"] > 300000)
    ][
        ["manufacturer", "model", "year", "mileage", "price"]
    ].head(50)
)

print(
    "Pre-1980 vehicles with <1,000 miles:",
    (
        (df["year"] < 1980) &
        (df["mileage"] < 1000)
    ).sum()
)

print(
    df[
        (df["year"] < 1980) &
        (df["mileage"] < 1000)
    ][
        ["manufacturer", "model", "year", "mileage", "price"]
    ].head(50)
)



print("\nSuspicious Price + Year")
print("_____________________________________________")


print(
    "2015+ vehicles priced below $2,000:",
    (
        (df["year"] >= 2015) &
        (df["price"] < 2000)
    ).sum()
)

print(
    df[
        (df["year"] >= 2015) &
        (df["price"] < 2000)
    ][
        ["manufacturer", "model", "year", "mileage", "price"]
    ].head(50)
)



print("\nBinary Columns")
print("_____________________________________________")


print("\naccidents_or_damage:")
print(df["accidents_or_damage"].value_counts(dropna=False))

print("\none_owner:")
print(df["one_owner"].value_counts(dropna=False))

print("\npersonal_use_only:")
print(df["personal_use_only"].value_counts(dropna=False))



print("\nCategorical Columns")
print("_____________________________________________")


for column in [
    "manufacturer",
    "model",
    "engine",
    "transmission",
    "drivetrain",
    "fuel_type",
    "mpg",
    "exterior_color",
    "interior_color",
    "seller_name"
]:
    print(
        column,
        "unique values:",
        df[column].nunique()
    )



print("\nMost Common Categories")
print("_____________________________________________")


print("\nManufacturers:")
print(df["manufacturer"].value_counts().head(20))

print("\nModels:")
print(df["model"].value_counts().head(20))

print("\nEngines:")
print(df["engine"].value_counts().head(20))

print("\nTransmissions:")
print(df["transmission"].value_counts().head(20))

print("\nDrivetrains:")
print(df["drivetrain"].value_counts().head(20))

print("\nFuel types:")
print(df["fuel_type"].value_counts().head(20))



print("\nMPG Analysis")
print("_____________________________________________")


print("MPG missing:", df["mpg"].isna().sum())
print("MPG unique values:", df["mpg"].nunique())

print("\nMost common MPG values:")
print(df["mpg"].value_counts().head(30))

print("\nExample MPG values:")
print(
    df["mpg"]
    .dropna()
    .drop_duplicates()
    .head(50)
)



print("\nRatings Analysis")
print("_____________________________________________")


print("\nSeller rating:")
print(df["seller_rating"].describe())

print(
    "Seller ratings outside 1-5:",
    (
        (df["seller_rating"] < 1) |
        (df["seller_rating"] > 5)
    ).sum()
)

print("\nDriver rating:")
print(df["driver_rating"].describe())

print(
    "Driver ratings outside 1-5:",
    (
        (df["driver_rating"] < 1) |
        (df["driver_rating"] > 5)
    ).sum()
)


print("\nPrice Drop Analysis")
print("_____________________________________________")


print(df["price_drop"].describe())

print(
    df["price_drop"].quantile(
        [0, .01, .05, .25, .50, .75, .95, .99, .999, 1]
    )
)

print("Price drop missing:", df["price_drop"].isna().sum())
print("Price drop = 0:", (df["price_drop"] == 0).sum())
print("Price drop > $1,000:", (df["price_drop"] > 1000).sum())
print("Price drop > $5,000:", (df["price_drop"] > 5000).sum())
print("Price drop > $10,000:", (df["price_drop"] > 10000).sum())

print("\nLargest price drops:")
print(
    df.nlargest(
        30,
        "price_drop"
    )[
        [
            "manufacturer",
            "model",
            "year",
            "price_drop",
            "price"
        ]
    ]
)


print("\nNumerical Correlations")
print("_____________________________________________")


print(
    df.select_dtypes(
        include=["int64", "float64"]
    ).corr()["price"].sort_values(
        ascending=False
    )
)
print("\n--- Cleaning Analysis Complete ---")
print("Original cars.csv was NOT modified.")