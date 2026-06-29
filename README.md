# Parcel Delivery Time Estimation using Linear Regression

**Author:** Deepa Aggarwal  
**Dataset:** Porter Food Delivery Orders (`porter_data_1.csv`)  
**Notebook:** `LR_Parcel_Delivery_Time_Estimation_Deepa_Aggarwal.ipynb`

---

## Project Overview

This project builds a **Multiple Linear Regression model** to predict the delivery time (in minutes) for food orders placed through **Porter** — an on-demand delivery platform. Accurate delivery time estimation helps optimize operational efficiency, improve customer satisfaction, and enable better resource allocation.

---

## Dataset Description

| Field | Description |
|---|---|
| `market_id` | Integer ID of the market where the restaurant is located |
| `created_at` | Timestamp when the order was placed |
| `actual_delivery_time` | Timestamp when the order was delivered |
| `store_primary_category` | Category of the restaurant (e.g., fast food) |
| `order_protocol` | Integer representing how the order was placed |
| `total_items` | Total number of items in the order |
| `subtotal` | Final price of the order |
| `num_distinct_items` | Number of distinct items in the order |
| `min_item_price` | Price of the cheapest item |
| `max_item_price` | Price of the most expensive item |
| `total_onshift_dashers` | Delivery partners on duty at order time |
| `total_busy_dashers` | Delivery partners already occupied |
| `total_outstanding_orders` | Orders pending fulfillment at order time |
| `distance` | Distance from restaurant to customer |

**Dataset Size:** 175,777 rows × 14 columns

---

## Project Pipeline

```
1. Data Loading
        ↓
2. Data Preprocessing & Feature Engineering
        ↓
3. Exploratory Data Analysis (EDA)
        ↓
4. Model Building (OLS Linear Regression)
        ↓
5. Model Inference & Evaluation
```

---

## Key Steps

### 1. Data Loading
- Loaded `porter_data_1.csv` into a Pandas DataFrame
- Inspected shape, datatypes, and basic statistics

### 2. Data Preprocessing & Feature Engineering
- **Datetime Conversion:** Converted `created_at` and `actual_delivery_time` from `object` to `datetime64`
- **Categorical Encoding:**
  - `market_id` and `order_protocol` — One-Hot Encoding (get_dummies, drop_first=True)
  - `store_primary_category` — Target Encoding (replaced with mean of `delivery_time_minutes` per category, due to 73 unique values)
- **Feature Engineering:**
  - `delivery_time_minutes` — Target variable: difference between actual delivery and order creation time (in minutes)
  - `hour` — Hour of day the order was placed
  - `day_of_week` — Day of week (0=Monday, 6=Sunday)
  - `is_Weekend` — Binary flag (1 if Saturday/Sunday)
- **Train-Test Split:** 70% train (123,043 rows) / 30% test (52,734 rows), `random_state=42`

### 3. Exploratory Data Analysis
- Visualized feature distributions (histograms, box plots)
- Correlation heatmap to identify relationships
- Outlier detection and removal using the **IQR method**
- Scatter plots between key features and target variable

### 4. Model Building
- Used `statsmodels.OLS` for interpretable regression with p-values and R-squared
- Iterative model building:
  - **Model 1:** Single feature — `distance` only
  - **Subsequent Models:** Added features incrementally based on statistical significance
  - **Final Model (Model 5):** 12 features — `distance`, `subtotal`, `total_items`, `total_onshift_dashers`, `total_busy_dashers`, `total_outstanding_orders`, `hour`, plus market_id and order_protocol dummies
- **Feature Selection:** p-value threshold (p < 0.05) and VIF analysis for multicollinearity
- **Feature Scaling:** StandardScaler / MinMaxScaler applied before model training

### 5. Model Evaluation
- **Metrics Used:** R-squared (R²), Adjusted R², RMSE (Root Mean Squared Error)
- **Residual Analysis:** Verified assumptions of linearity, homoscedasticity, and normality of errors

---

## Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical computations |
| `matplotlib` | Data visualization |
| `seaborn` | Statistical plots |
| `statsmodels` | OLS regression with statistical summaries |
| `sklearn` | Train-test split, scaling, metrics |

---

## Key Findings

- **Distance** is the most significant predictor of delivery time
- **`total_outstanding_orders`** has a strong positive effect on delivery time (more pending orders = longer wait)
- **`total_onshift_dashers`** has a negative effect (more available dashers = faster delivery)
- **Hour of day** influences delivery time (negative coefficient — later hours have different patterns)
- Market location (market_id dummies) significantly affects delivery time

---

## How to Run

1. Clone or download this repository
2. Ensure the dataset `porter_data_1.csv` is available at the correct path
3. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn statsmodels scikit-learn
   ```
4. Open and run the notebook:
   ```bash
   jupyter notebook LR_Parcel_Delivery_Time_Estimation_Deepa_Aggarwal.ipynb
   ```

---

## Project Structure

```
LR_Delivery_Time_Prediction_Deepa_Aggarwal/
│
├── LR_Parcel_Delivery_Time_Estimation_Deepa_Aggarwal.ipynb   # Main notebook
├── README.md                                                   # Project documentation
└── porter_data_1.csv                                          # Dataset (not included in repo)
```
