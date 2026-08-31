# 🏦 Customer Credit Risk Data Preprocessing & Feature Engineering

## 📌 Project Overview

This project demonstrates an end-to-end **Data Preprocessing and Feature Engineering workflow** using a Customer Credit Risk dataset.

The data is collected from multiple sources including:

- CSV
- JSON
- MySQL Database

The objective of this project is to collect, merge, clean, transform, encode, scale, and engineer features from raw customer credit data.

The final processed dataset is prepared for future **Machine Learning Binary Classification** to predict whether a customer is likely to default on a loan.

---

# 🎯 Project Objective

The project includes:

- Data collection from multiple sources
- Data integration and merging
- Data understanding using Pandas
- Data profiling
- Missing value handling
- Outlier detection and treatment
- Date and time feature extraction
- Categorical feature encoding
- Numerical feature encoding
- Feature scaling
- Data transformations
- Feature engineering
- Final cleaned dataset preparation

---

# 🤖 Machine Learning Problem

This project can be framed as a **Binary Classification problem**.

## Target Variable: `default_flag`

| Value | Meaning |
|---|---|
| 0 | Customer did not default |
| 1 | Customer defaulted on a loan |

A future Machine Learning model can use customer demographics, financial information, loan details, transaction behavior, and engineered features to predict loan default.

---

# 📂 Data Sources

## 1️⃣ CSV Data

**File:** `customers.csv`

Contains:

- customer_id
- age
- gender
- region
- education_level
- employment_type
- annual_income
- join_date
- default_flag

## 2️⃣ JSON Data

**File:** `transactions.json`

Contains:

- transaction_id
- customer_id
- product_id
- loan_amount
- loan_purpose
- credit_score
- repayment_months
- transaction_count
- interest_rate

## 3️⃣ MySQL Data

**Database:** `customer_credit_db`

**Table:** `products`

Contains:

- product_id
- product_name
- category
- interest_rate
- stock

---

# 🔗 Data Integration

The datasets are merged using:

- `customer_id`
- `product_id`

```python
df = pd.merge(
    customers,
    transactions,
    on="customer_id",
    how="left"
)

products = products.rename(
    columns={
        "interest_rate": "product_interest_rate"
    }
)

df = pd.merge(
    df,
    products,
    on="product_id",
    how="left"
)
```

---

# 🛠️ Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- MySQL
- MySQL Workbench
- MySQL Connector
- YData Profiling

---

# 📦 Required Libraries

```bash
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install scipy
pip install mysql-connector-python
pip install ydata-profiling
```

---

# 📥 Data Loading

## Import Libraries

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

## Load CSV Data

```python
customers = pd.read_csv("customers.csv")
customers.head()
```

## Load JSON Data

```python
transactions = pd.read_json("transactions.json")
transactions.head()
```

## Connect to MySQL

```python
import mysql.connector

connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="YOUR_PASSWORD",
    database="customer_credit_db"
)

print("MySQL Connected Successfully!")
```

## Load MySQL Table

```python
products = pd.read_sql(
    "SELECT * FROM products",
    connection
)

products.head()
```

---

# 🔍 Data Understanding

```python
print("Dataset Shape:", df.shape)

display(df.head())

df.info()

display(df.describe(include="all"))

print(df.dtypes)
```

---

# 📊 Data Profiling

```python
from ydata_profiling import ProfileReport

profile = ProfileReport(
    df,
    title="Customer Credit Risk Data Profiling Report",
    explorative=True
)

profile.to_file(
    "customer_credit_profile_report.html"
)
```

---

# 🧹 Missing Value Handling

## Check Missing Values

```python
missing_values = df.isnull().sum()

display(
    missing_values[
        missing_values > 0
    ].sort_values(ascending=False)
)
```

## Simple Imputer

```python
from sklearn.impute import SimpleImputer

numerical_columns = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score"
]

median_imputer = SimpleImputer(strategy="median")

df[numerical_columns] = median_imputer.fit_transform(
    df[numerical_columns]
)
```

## Most Frequent Imputation

```python
categorical_columns = [
    "gender",
    "employment_type"
]

mode_imputer = SimpleImputer(
    strategy="most_frequent"
)

df[categorical_columns] = mode_imputer.fit_transform(
    df[categorical_columns]
)
```

## Missing Indicator

```python
for column in numerical_columns:
    df[column + "_missing"] = (
        df[column].isnull().astype(int)
    )
```

## Random Sample Imputation

```python
df_random = df.copy()

column = "annual_income"

missing = df_random[column].isnull()

random_values = (
    df_random[column]
    .dropna()
    .sample(
        missing.sum(),
        replace=True,
        random_state=42
    )
    .values
)

df_random.loc[missing, column] = random_values
```

## KNN Imputer

```python
from sklearn.impute import KNNImputer

df_knn = df.copy()

knn_imputer = KNNImputer(n_neighbors=5)

df_knn[numerical_columns] = knn_imputer.fit_transform(
    df_knn[numerical_columns]
)
```

## MICE / Iterative Imputer

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

df_mice = df.copy()

mice_imputer = IterativeImputer(
    random_state=42
)

df_mice[numerical_columns] = mice_imputer.fit_transform(
    df_mice[numerical_columns]
)
```

## Complete Case Analysis

```python
df_complete_case = df.dropna()

print("Original Shape:", df.shape)
print("Complete Case Shape:", df_complete_case.shape)
```

---

# 📈 Outlier Detection

## Z-Score Method

```python
from scipy.stats import zscore

outlier_columns = [
    "annual_income",
    "loan_amount",
    "credit_score"
]

for column in outlier_columns:

    z_scores = np.abs(
        zscore(df[column].dropna())
    )

    print(
        column,
        "Outliers:",
        (z_scores > 3).sum()
    )
```

## IQR Method

```python
for column in outlier_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(column, "Outliers:", len(outliers))
```

## Percentile Method

```python
for column in outlier_columns:

    lower_limit = df[column].quantile(0.01)
    upper_limit = df[column].quantile(0.99)

    outliers = df[
        (df[column] < lower_limit) |
        (df[column] > upper_limit)
    ]

    print(column, "Outliers:", len(outliers))
```

## Winsorization

```python
df_winsorized = df.copy()

for column in outlier_columns:

    lower_limit = df_winsorized[column].quantile(0.01)
    upper_limit = df_winsorized[column].quantile(0.99)

    df_winsorized[column] = np.clip(
        df_winsorized[column],
        lower_limit,
        upper_limit
    )
```

---

# 📅 Date Feature Engineering

```python
df["join_date"] = pd.to_datetime(
    df["join_date"],
    errors="coerce"
)

df["join_year"] = df["join_date"].dt.year
df["join_month"] = df["join_date"].dt.month
df["join_day"] = df["join_date"].dt.day
df["join_weekday"] = df["join_date"].dt.day_name()
```

---

# 🔤 Categorical Encoding

## Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder

education_order = [[
    "Secondary",
    "Graduate",
    "Post-Graduate"
]]

ordinal_encoder = OrdinalEncoder(
    categories=education_order
)

df["education_level_encoded"] = (
    ordinal_encoder.fit_transform(
        df[["education_level"]]
    )
)
```

## Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

df["gender_encoded"] = (
    label_encoder.fit_transform(
        df["gender"].astype(str)
    )
)
```

## One-Hot Encoding

```python
df_encoded = pd.get_dummies(
    df,
    columns=[
        "region",
        "loan_purpose"
    ],
    drop_first=True,
    dtype=int
)
```

---

# 🔢 Numerical Feature Encoding

## Binning

```python
df["income_category"] = pd.qcut(
    df["annual_income"],
    q=4,
    labels=[
        "Low",
        "Medium",
        "High",
        "Very High"
    ],
    duplicates="drop"
)
```

## Binarization

```python
df["good_credit"] = np.where(
    df["credit_score"] > 700,
    1,
    0
)
```

## Quantile Binning

```python
from sklearn.preprocessing import KBinsDiscretizer

quantile_binner = KBinsDiscretizer(
    n_bins=4,
    encode="ordinal",
    strategy="quantile"
)

df["transaction_bin"] = (
    quantile_binner.fit_transform(
        df[["transaction_count"]]
    )
)
```

## K-Means Binning

```python
from sklearn.cluster import KMeans

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["transaction_cluster"] = (
    kmeans.fit_predict(
        df[["transaction_count"]]
    )
)
```

---

# 📏 Feature Scaling

## StandardScaler

```python
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

df["income_standard"] = (
    standard_scaler.fit_transform(
        df[["annual_income"]]
    )
)
```

## MinMaxScaler

```python
from sklearn.preprocessing import MinMaxScaler

minmax_scaler = MinMaxScaler()

df["income_minmax"] = (
    minmax_scaler.fit_transform(
        df[["annual_income"]]
    )
)
```

## MaxAbsScaler

```python
from sklearn.preprocessing import MaxAbsScaler

maxabs_scaler = MaxAbsScaler()

df["loan_maxabs"] = (
    maxabs_scaler.fit_transform(
        df[["loan_amount"]]
    )
)
```

## RobustScaler

```python
from sklearn.preprocessing import RobustScaler

robust_scaler = RobustScaler()

df["income_robust"] = (
    robust_scaler.fit_transform(
        df[["annual_income"]]
    )
)
```

## Normalizer

```python
from sklearn.preprocessing import Normalizer

normalizer = Normalizer()

normalized_data = normalizer.fit_transform(
    df[
        [
            "annual_income",
            "loan_amount"
        ]
    ]
)

normalized_df = pd.DataFrame(
    normalized_data,
    columns=[
        "annual_income_normalized",
        "loan_amount_normalized"
    ]
)
```

---

# 🔄 Data Transformations

## Log Transformation

```python
df["spending_ratio"] = (
    df["loan_amount"] /
    df["annual_income"]
) * 100

df["spending_log"] = np.log1p(
    df["spending_ratio"]
)
```

## Reciprocal Transformation

```python
df["spending_reciprocal"] = (
    1 / (df["spending_ratio"] + 1)
)
```

## Square Root Transformation

```python
df["spending_sqrt"] = np.sqrt(
    df["spending_ratio"].clip(lower=0)
)
```

## Box-Cox Transformation

```python
from sklearn.preprocessing import PowerTransformer

boxcox_transformer = PowerTransformer(
    method="box-cox"
)

positive_income = (
    df[["annual_income"]]
    .dropna()
)

boxcox_values = (
    boxcox_transformer.fit_transform(
        positive_income
    )
)
```

## Yeo-Johnson Transformation

```python
yeojohnson_transformer = PowerTransformer(
    method="yeo-johnson"
)

df["loan_yeojohnson"] = (
    yeojohnson_transformer.fit_transform(
        df[["loan_amount"]]
    )
)
```

---

# ⚙️ ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

numerical_columns = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "transaction_count"
]

categorical_columns = [
    "gender",
    "employment_type",
    "region"
]

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numerical_columns
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        )
    ]
)

processed_data = (
    preprocessor.fit_transform(df)
)

print(processed_data.shape)
```

---

# 🏗️ Feature Engineering

## Debt-to-Income Ratio

```python
df["debt_to_income_ratio"] = (
    df["loan_amount"] /
    df["annual_income"]
)
```

## Average Monthly Transactions

```python
df["months_since_join"] = (
    (
        pd.Timestamp.today() -
        df["join_date"]
    ).dt.days / 30
).clip(lower=1)

df["average_monthly_transactions"] = (
    df["transaction_count"] /
    df["months_since_join"]
)
```

## Spending-to-Income Ratio

```python
df["spending_to_income_ratio"] = (
    df["loan_amount"] /
    df["annual_income"]
)
```

---

# 💾 Final Dataset

```python
final_df = df.copy()

final_df = final_df.drop_duplicates()

print("Final Dataset Shape:", final_df.shape)

print(
    "Total Missing Values:",
    final_df.isnull().sum().sum()
)
```

## Save Final Dataset

```python
final_df.to_csv(
    "final_customer_credit_dataset.csv",
    index=False
)

print("Final dataset saved successfully!")
```

---

# 🔄 Complete Project Workflow

```text
Data Collection
       ↓
CSV + JSON + MySQL
       ↓
Data Integration
       ↓
Data Understanding
       ↓
Data Profiling
       ↓
Missing Value Handling
       ↓
Outlier Detection
       ↓
Date Feature Engineering
       ↓
Categorical Encoding
       ↓
Numerical Encoding
       ↓
Feature Scaling
       ↓
Data Transformation
       ↓
Feature Engineering
       ↓
Final Cleaned Dataset
       ↓
Ready for Machine Learning
```

---

# 📁 Project Structure

```text
Customer-Credit-Risk-Data-Preprocessing/
│
├── customers.csv
├── transactions.json
├── products.sql
├── Customer_Credit_Risk_Project.ipynb
├── Holistic_Data_Summary.pdf
├── customer_credit_profile_report.html
├── final_customer_credit_dataset.csv
└── README.md
```

---

# 📚 Key Techniques Used

### Missing Value Handling

- Simple Imputer
- Most Frequent Imputation
- Missing Indicator
- Random Sample Imputation
- KNN Imputer
- MICE / Iterative Imputer
- Complete Case Analysis

### Outlier Detection

- Z-Score
- IQR Method
- Percentile Method
- Winsorization

### Encoding

- Ordinal Encoding
- Label Encoding
- One-Hot Encoding

### Numerical Encoding

- Binning
- Binarization
- Quantile Binning
- K-Means Binning

### Scaling

- StandardScaler
- MinMaxScaler
- MaxAbsScaler
- RobustScaler
- Normalizer

### Transformations

- Log Transformation
- Reciprocal Transformation
- Square Root Transformation
- Box-Cox Transformation
- Yeo-Johnson Transformation
- PowerTransformer
- ColumnTransformer

---

# 🎓 Key Learnings

Through this project, I learned how to:

- Collect data from multiple sources
- Work with CSV, JSON, and MySQL data
- Merge multiple datasets
- Generate data profiling reports
- Handle missing values using multiple techniques
- Detect and treat outliers
- Encode categorical variables
- Scale numerical features
- Transform skewed data
- Create meaningful new features
- Build an end-to-end data preprocessing workflow

---

# 🚀 Final Outcome

The final dataset was:

- Collected from multiple sources
- Merged into a single dataset
- Cleaned and preprocessed
- Checked for missing values
- Treated for outliers
- Encoded
- Scaled
- Transformed
- Feature-engineered

The final processed dataset is ready for future **Machine Learning Binary Classification**.

## Prediction Target

```text
default_flag

0 = No Default
1 = Default
```

---

# 👨‍💻 Author

**Mahesh Lohar**

---

## ⭐ Support

If you found this project useful, please consider giving this repository a **Star ⭐**!
