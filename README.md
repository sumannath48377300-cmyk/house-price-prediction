# 🏠 House Price Prediction & Analytics

A production-ready web application built with **Python**, **Streamlit**, and **Machine Learning** for predicting and analysing house prices using the `Housing_cleaned.csv` dataset.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [App Pages](#app-pages)
- [Screenshots](#screenshots)
- [Tech Stack](#tech-stack)

---

## Overview

This app allows users to:
- Explore housing data through rich interactive visualisations
- Compare the performance of 7 machine learning models
- Predict house prices by entering property details
- Analyse which features drive house prices the most
- Filter, sort, and download the full dataset

---

## Features

- 📊 **Interactive Dashboard** — KPIs, price distributions, and model comparison at a glance
- 🔍 **Data Explorer** — Histograms, scatter plots, box plots, and a full correlation heatmap
- 🤖 **Model Performance** — Side-by-side metrics, actual vs predicted charts, and residual analysis
- 🔮 **Price Predictor** — Real-time price estimation with comparable property lookup
- 📈 **Feature Insights** — Feature importance, binary feature price premiums, and what-if sensitivity analysis
- 📋 **Data Table** — Filterable dataset with CSV download

---

## Dataset

**File:** `Housing_cleaned.csv`  
**Rows:** 545 properties  
**Features:** 13 columns

| Column | Type | Description |
|---|---|---|
| `price` | int | Sale price of the house (₹) |
| `area` | int | Area in square feet |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | int | Number of bathrooms |
| `stories` | int | Number of stories |
| `parking` | int | Number of parking spaces |
| `mainroad` | binary | Located on main road (1 = Yes) |
| `guestroom` | binary | Has guest room (1 = Yes) |
| `basement` | binary | Has basement (1 = Yes) |
| `hotwaterheating` | binary | Hot water heating available (1 = Yes) |
| `airconditioning` | binary | Air conditioning available (1 = Yes) |
| `prefarea` | binary | Located in preferred area (1 = Yes) |
| `furnishingstatus` | int | 0 = Unfurnished, 1 = Semi-Furnished, 2 = Furnished |

---

## Machine Learning Models

Seven regression models are trained and evaluated on every run:

| Model | Notes |
|---|---|
| Linear Regression | Baseline linear model, trained on scaled features |
| Ridge Regression | L2-regularised linear model (α = 10) |
| Lasso Regression | L1-regularised linear model (α = 1000) |
| Decision Tree | Depth-limited to 6 levels |
| Random Forest | 200 estimators, max depth 10 |
| Gradient Boosting | 200 estimators, learning rate 0.05 |
| XGBoost | 200 estimators, learning rate 0.05 (optional) |

### Evaluation Metrics

Each model is assessed using:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination
- **MAPE** — Mean Absolute Percentage Error
- **5-Fold Cross-Validated R²** (mean ± std)

The best model is automatically selected by test R² and highlighted throughout the app.

---

## Project Structure

```
House Price Prediction/
├── app.py                  # Main Streamlit application
├── Housing_cleaned.csv     # Dataset
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## Installation

### 1. Clone or download the project

```bash
git clone <your-repo-url>
cd "House Price Prediction"
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** XGBoost is optional. If installation fails, the app falls back gracefully to the remaining 6 models.

---

## Usage

```bash
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## App Pages

### 🏠 Dashboard
High-level overview of the dataset and model results:
- Key metrics (total properties, avg/median price, avg area, best R²)
- Price distribution histogram
- Properties by furnishing status (donut chart)
- Area vs price scatter coloured by furnishing
- Average price by number of bedrooms
- Model R² comparison bar chart

### 📊 Data Explorer
Deep-dive into the data across 4 tabs:
- **Distributions** — Histogram with marginal box plot for any feature; mean, median, std, range stats
- **Correlations** — Scatter plot with OLS trendline; customisable axes and colour grouping
- **Box Plots** — Price distribution grouped by any categorical feature
- **Heat Map** — Full Pearson correlation matrix for all numeric features

### 🤖 Model Performance
Comprehensive model evaluation:
- Colour-highlighted metrics table (best values in each column highlighted)
- R² and MAPE bar charts for all models
- Actual vs Predicted scatter with perfect-fit reference line
- Residual distribution histogram and residuals vs predicted scatter

### 🔮 Price Predictor
Input any property configuration and get an instant price estimate:
- Sliders and dropdowns for all 12 input features
- Predicted price formatted in ₹ Lakh or ₹ Crore
- Price per sq ft
- ±10% confidence range
- Comparable properties from the dataset

### 📈 Feature Insights
Understand what drives house prices:
- **Feature Importance** — Bar chart from tree-based models or permutation importance
- **Price Drivers** — Grouped bar chart showing avg price with/without each binary feature; price premium table
- **What-If Analysis** — Line chart showing how predicted price changes as a single feature varies

### 📋 Data Table
Browse and filter the full dataset:
- Filter by price range (slider), bedrooms (multiselect), and furnishing status
- Summary statistics for filtered results
- One-click CSV download of filtered data

---

## Tech Stack

| Component | Library / Version |
|---|---|
| Web framework | Streamlit ≥ 1.32 |
| Data processing | Pandas ≥ 2.0, NumPy ≥ 1.24 |
| Machine learning | scikit-learn ≥ 1.3, XGBoost ≥ 2.0 |
| Visualisation | Plotly ≥ 5.18 |
| Model persistence | joblib ≥ 1.3 |
| Language | Python 3.9+ |

---

## License

This project is for educational and demonstration purposes.
