# CarAnalyzer 🚗

A data science project focused on analyzing used-car listings, understanding the factors that influence vehicle prices, and eventually building a tool to determine whether a car is a good value.

## Project Goal

CarAnalyzer will use a large used-car dataset to:

* Clean and validate vehicle listing data
* Explore relationships between vehicle characteristics and price
* Identify the factors most associated with vehicle value
* Build a machine learning model to estimate market prices
* Eventually create a tool that evaluates whether a specific vehicle is priced fairly

## Current Progress

### Phase 1 — Data Cleaning ✅

The original dataset contains over **760,000 used-car listings**.

The first phase focused on identifying and handling data-quality issues, including:

* Removing exact duplicate listings
* Removing unrealistic vehicle prices
* Removing implausible mileage values
* Handling zero-mileage values that likely represent missing data
* Flagging unusually low-mileage classic vehicles instead of removing them
* Splitting MPG into city and highway values
* Standardizing drivetrain categories
* Standardizing fuel-type categories
* Converting binary variables into boolean values
* Creating a price-drop indicator

The original dataset is preserved, while the cleaned dataset is saved separately.

## Project Structure

```text
CarAnalyzer/
│
├── cars.csv
├── cars_clean.csv
├── clean_cars_data.py
└── README.md
```

## Tech Stack

Currently:

* Python
* pandas
* NumPy
* Git/GitHub

Planned:

* SQL
* Snowflake
* scikit-learn
* PyTorch
* AWS
* Streamlit

## Upcoming Phases

### Phase 2 — Exploratory Data Analysis

Analyze relationships between price and variables such as:

* Vehicle age
* Mileage
* Manufacturer
* Model
* Drivetrain
* Fuel type
* Accident history
* Ownership history

### Phase 3 — Data Warehouse & SQL

Load the cleaned dataset into Snowflake and use SQL to perform larger-scale analysis.

### Phase 4 — Machine Learning

Build and compare models that predict the expected market listing price of a vehicle.

### Phase 5 — Car Value Analyzer

Use the predicted price and actual listing price to determine whether a vehicle appears to be:

* Excellent value
* Good value
* Fair value
* Expensive
* Poor value

### Phase 6 — Application

Build a simple web application where users can enter vehicle information and receive an estimated market value and value assessment.
