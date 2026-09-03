"""
analysis.py

Compute some statistics about the values and find relationships between variables
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("cars_clean.csv")

print(df.shape)
print(df.describe())
print(df.dtypes)

#Year of the dataset is 2024
current_year = 2024
df["vehicle_age"] = current_year - df["year"]
df["mileage_per_year"] = df["mileage"] / df["vehicle_age"].replace(0,1)

#Discover: Listing price, price skew, price variation, and if they are from luxury vehicles.
print(df["price"].describe())

plt.hist(df["price"], bins=100)
plt.xlabel("Price")
plt.ylabel("Number of Vehicles")
plt.title("Distribution of Vehicle Prices")
plt.show()

plt.scatter(df["vehicle_age"], df["price"], alpha=0.1)
plt.xlabel("Vehicle Age")
plt.ylabel("Price")
plt.title("Vehicle Age vs. Listing price")
plt.show()

print(f"Correlation between vehicle age and price: {df["vehicle_age"].corr(df["price"])}")

plt.scatter(df["mileage"], df["price"], alpha=0.1)
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Mileage vs. Listing price")
plt.show()

print(f"Correlation between mileage and price: {df["mileage"].corr(df["price"])}")

manufact_price = (
    df.groupby("manufacturer")["price"]
    .agg(["count", "mean", "median"])
    .sort_values("median", ascending=False)
)

print(manufact_price)

top_manufact = (
    df["manufacturer"].value_counts().head(15).index
)

manufact_data = df[df["manufacturer"].isin(top_manufact)]

manufact_data.boxplot(column="price", by="manufacturer", rot=90)

plt.ylabel("Price")
plt.title("Vehicle Price by manufacturer")
plt.suptitle("")
plt.show()

#Model vs price
model_counts = df["model"].value_counts()

common_models = model_counts[model_counts >= 500].index

model_price = (
    df[df["model"].isin(common_models)]
    .groupby("model")["price"]
    .agg(["count", "mean", "median"])
    .sort_values("median", ascending=False)
)
print(model_price)

#Drivetrain vs price
print(df.groupby("drivetrain")["price"].agg(["count", "mean", "median"]).sort_values("median", ascending=False))

#Fuel type vs price
fuel_price = (df.groupby("fuel_type")["price"]
        .agg(["count", "mean", "median"])
        .sort_values("median", ascending=False))
print(fuel_price)


print(df["accidents_or_damage"].value_counts(dropna=False))
print(df.groupby("accidents_or_damage")["price"].agg(["count", "mean", "median"]))

print(df["accidents_or_damage"].value_counts(dropna=False))
print(
    df.groupby("accidents_or_damage")["price"]
    .agg(["count", "mean", "median"])
)

print(
    df.groupby("one_owner")["price"]
    .agg(["count", "mean", "median"])
)

print(
    df.groupby("personal_use_only")["price"]
    .agg(["count", "mean", "median"])
)

plt.scatter(
    df["vehicle_age"],
    df["mileage"],
    alpha=0.1
)

plt.xlabel("Vehicle Age")
plt.ylabel("Mileage")
plt.title("Vehicle Age vs. Mileage")
plt.show()

numeric_columns = [
    "price",
    "year",
    "vehicle_age",
    "mileage",
    "mileage_per_year",
    "seller_rating",
    "driver_rating",
    "driver_reviews_num"
]

print(df[numeric_columns].corr()["price"].sort_values())