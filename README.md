# 📊 Sales Prediction Analysis using Machine Learning

A machine learning project focused on analysing retail sales data and predicting **Profit** using supervised regression techniques. The project implements a complete data science workflow, including data cleaning, exploratory data analysis, feature engineering, preprocessing, model training, evaluation, cross-validation, hyperparameter tuning, and SHAP-based model interpretation.

---

## 📌 Project Overview

This project analyses a synthetic Superstore-style retail dataset containing **10,000 sales orders** recorded between **January 2023 and December 2024**.

The primary objective was to investigate whether historical sales information could be used to reliably predict:

* 💰 Revenue
* 📈 Profit

An important characteristic of the dataset was discovered during analysis: **Revenue is deterministically calculated as Quantity × Unit Price**. Therefore, Quantity and Unit Price were excluded from the Revenue prediction model to avoid creating a trivial prediction problem.

Profit, on the other hand, was found to be a genuinely learnable target and became the primary focus of the machine learning modelling process.

---

## 🎯 Objectives

The major objectives of this project are:

* Analyse historical retail sales data.
* Clean and preprocess the raw dataset.
* Perform exploratory and diagnostic data analysis.
* Identify important patterns and relationships in sales and profit.
* Engineer useful temporal and categorical features.
* Build a leakage-free machine learning pipeline.
* Compare Linear Regression, Random Forest, and XGBoost.
* Evaluate models using R², MAE, and RMSE.
* Perform time-series cross-validation.
* Perform hyperparameter tuning.
* Interpret the best-performing model using SHAP.
* Generate business-oriented insights from the model results.

---

## 📂 Dataset

The project uses a **synthetic Superstore-style retail dataset** obtained from publicly available Kaggle sources.

### Dataset Characteristics

| Property         | Details                      |
| ---------------- | ---------------------------- |
| Records          | 10,000 orders                |
| Columns          | 14                           |
| Time Period      | January 2023 – December 2024 |
| Categories       | 4                            |
| Sub-Categories   | 19                           |
| Products         | 49                           |
| Regions          | 4                            |
| Target Variables | Revenue, Profit              |

The original dataset contains order information, customer information, geographical attributes, product details, transaction quantities and prices, Revenue, and Profit.

---

## 🧹 Data Preprocessing

The preprocessing workflow included:

* Loading the dataset using Pandas.
* Inspecting dataset structure and data types.
* Converting mixed-format date values.
* Removing non-predictive identifier columns.
* Handling constant and near-unique features.
* Creating temporal features such as:

  * Year
  * Month
  * Quarter
* One-hot encoding categorical variables.
* Scaling numerical variables for Linear Regression.
* Keeping tree-based model features unscaled.
* Handling unseen categorical values using `handle_unknown='ignore'`.

A `Pipeline` and `ColumnTransformer` architecture was used to ensure that preprocessing transformations were fitted only on the training data, preventing data leakage.

---

## ⏳ Train-Test Split

Instead of using a random train-test split, the project uses a **chronological time-based split**.

| Dataset  | Period              | Records |
| -------- | ------------------- | ------: |
| Training | Jan 2023 – Oct 2024 |   8,317 |
| Testing  | Nov 2024 – Dec 2024 |   1,683 |

The test set deliberately contains the November–December period because November showed a strong seasonal sales and profit peak.

This approach helps prevent temporal data leakage and provides a more realistic estimate of future model performance.

---

## 🤖 Machine Learning Models

Three regression approaches were evaluated:

### 1. Linear Regression

Used as the baseline model because it is simple, interpretable, and provides a reference point for evaluating more complex models.

### 2. Random Forest Regressor

A tree-based ensemble model capable of capturing nonlinear relationships and feature interactions.

The project identified Random Forest as the **best-performing model for Profit prediction**.

### 3. XGBoost Regressor

A gradient boosting algorithm that sequentially builds trees to reduce prediction errors.

---

## 📊 Model Performance

### Revenue Prediction

| Model             |    R² |     MAE |    RMSE |
| ----------------- | ----: | ------: | ------: |
| Linear Regression | 0.264 | $426.50 | $634.45 |
| Random Forest     | 0.214 | $438.60 | $655.40 |
| XGBoost           | 0.140 | $451.16 | $685.51 |

The results show that Revenue could not be reliably predicted using the available non-formulaic features after excluding Quantity and Unit Price.

### Profit Prediction

| Model             |        R² |        MAE |       RMSE |
| ----------------- | --------: | ---------: | ---------: |
| Linear Regression |     0.795 |     $45.82 |     $68.40 |
| Random Forest ⭐   | **0.860** | **$36.35** | **$56.55** |
| XGBoost           |     0.855 |     $36.43 |     $57.49 |

Random Forest achieved the best overall performance for Profit prediction, improving the R² score from **0.795 to 0.860** compared with the Linear Regression baseline.

---

## 🏆 Best Model

**Random Forest Regressor for Profit Prediction**

### Performance

* **R²:** 0.860
* **MAE:** $36.35
* **RMSE:** $56.55

The model was further validated using five-fold time-series cross-validation, achieving a mean R² of approximately **0.847** with a standard deviation of **0.016**, indicating stable performance across different temporal windows.

---

## 🔍 Model Interpretability

SHAP (SHapley Additive exPlanations) was used to understand the factors influencing the Random Forest Profit predictions.

Important predictive factors identified include:

* Revenue
* Product Category
* Quantity
* Unit Price
* Temporal characteristics

The analysis showed that **Product Category** has a significant influence on predicted Profit.

In particular, Electronics orders were associated with lower predicted profit relative to the dataset average, while categories such as Accessories and Clothing & Apparel showed comparatively stronger profitability.

---

## 🛠️ Technologies Used

### Programming Language

* Python 3.11

### Data Processing

* Pandas
* NumPy

### Machine Learning

* Scikit-learn
* XGBoost

### Data Visualization

* Matplotlib
* Seaborn

### Model Interpretation

* SHAP

### Model Persistence & Data Exchange

* Joblib
* JSON

The report specifically identifies these tools as the core technologies used throughout the modelling, validation, visualization, interpretability, and model-persistence stages.

---

## 🔄 Project Workflow

```text
Raw Dataset
     │
     ▼
Data Loading
     │
     ▼
Data Cleaning
     │
     ▼
Date Conversion & Feature Engineering
     │
     ▼
Exploratory Data Analysis
     │
     ▼
Target Analysis
 ┌───┴─────────────┐
 │                 │
Revenue           Profit
 │                 │
 ▼                 ▼
Feature Selection  Feature Selection
 │                 │
 └───────┬─────────┘
         ▼
Time-Based Train/Test Split
         │
         ▼
Preprocessing Pipeline
         │
         ▼
Model Training
 ┌───────┼──────────┐
 │       │          │
 ▼       ▼          ▼
Linear  Random    XGBoost
Reg.    Forest
         │
         ▼
Model Evaluation
         │
         ▼
Cross-Validation
         │
         ▼
Hyperparameter Tuning
         │
         ▼
SHAP Interpretability
         │
         ▼
Business Insights
```

---

## 📈 Key Findings

### Revenue

Revenue prediction using categorical, regional, and temporal features alone produced relatively weak results.

This is primarily because:

```text
Revenue = Quantity × Unit Price
```

Therefore, Revenue is fundamentally a derived value rather than an independently learnable outcome in this dataset.

### Profit

Profit proved to be a much more suitable machine learning target.

The Random Forest model achieved:

> **R² = 0.860**

with an average absolute prediction error of:

> **MAE = $36.35**

The model also demonstrated stable performance through time-series cross-validation.

---

## 💡 Business Insights

The project provides several useful business insights:

* Product category has a major influence on profitability.
* Electronics showed comparatively weaker profit margins.
* Accessories demonstrated stronger profitability.
* Profit varies across product categories and time periods.
* Seasonal effects, particularly around November, influence sales and profit.
* Predictive modelling can help identify potentially low-profit orders or product categories.
* Model interpretability makes the predictions more useful for business decision-making.

---

## 🔮 Future Scope

Potential future improvements include:

### 1. Additional Machine Learning Models

Future versions can experiment with:

* LightGBM
* CatBoost
* Stacking and ensemble approaches

### 2. Real-World Deployment

The trained Profit model could be deployed into an operational forecasting system with continuous monitoring for model drift.

### 3. Prescriptive Analytics

SHAP explanations could be converted into automated recommendations, such as identifying upcoming orders or products likely to fall below a desired profit margin.

### 4. Richer Data

Additional variables could improve prediction performance, including:

* Cost of goods sold
* Marketing expenditure
* Competitor pricing
* Macroeconomic indicators
* Additional customer-level information

These additions could help explain variance that remains unexplained by the current model.

---

## 🎓 Academic Information

**Project:** Sales Prediction Analysis using Machine Learning

**Student:** Anuj Kumar

**Programme:** B.Tech – Computer Science and Engineering

**Internship at:** B.I.T. Sindri

**Project Guide:** Mr. Jitendra Kumar, Assistant Professor, B.I.T SINDRI

**Academic Session:** 2025–2026

**Report Date:** August 2026

---

## 📜 Conclusion

This project demonstrates an end-to-end machine learning workflow for retail sales analysis and Profit prediction.

The most important outcome was not simply achieving a high model score, but correctly identifying the structural relationship within the dataset before modelling. Revenue was found to be a deterministic calculation, while Profit was shown to be a genuinely learnable target.

The final Random Forest model achieved **R² = 0.860** and **MAE = $36.35**, providing a validated and interpretable approach for Profit prediction. SHAP analysis further helped translate model predictions into actionable business insights.

---

## 👨‍💻 Author

**Anuj Kumar**

B.Tech Computer Science and Engineering
IILM UNIVERSITY, GREATER NOIDA

---

⭐ If you find this project useful, consider giving the repository a star!
