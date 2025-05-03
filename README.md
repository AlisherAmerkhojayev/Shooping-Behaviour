# Customer Shopping Behavior Analysis in Online Retail

---

##  Personal Motivation

This project was born out of a desire to bridge **analytical rigor with business decision-making**. As an aspiring data analyst in retail and finance, I sought to replicate a real-world problem where customer data becomes the foundation for **personalized marketing**, **subscription retention**, and **strategic segmentation**. The ability to distill high-dimensional behavioral data into insights that directly influence **customer lifetime value (CLV)**, marketing ROI, and customer engagement is a powerful asset in today’s data-first enterprises.

---

## Project Summary

The project applies a combination of **unsupervised and supervised machine learning techniques** to online retail customer data. The goal was to:

1. Segment customers into meaningful behavioral cohorts
2. Identify promotion-engaged individuals suitable for campaign targeting
3. Predict subscription status using demographic and behavioral cues

The approach emphasizes **model selection**, **feature engineering**, and **interpretable metrics**, aiming to emulate a full analytics pipeline from data audit to business insight.

---

## Dataset Overview

- **Source**: [Kaggle - Consumer Behavior and Shopping Habits](https://www.kaggle.com/datasets/zeesolver/consumer-behavior-and-shopping-habits-dataset)
- **Shape**: 3,901 rows × 19 columns
- **Type**: Mixed-type (categorical + numerical)
- **Targets**:
  - Clustering labels (from KMeans)
  - `Target Customer` (custom binary classification)
  - `Subscription Status` (existing binary column)

The dataset offers a comprehensive profile of each customer, combining static demographic traits (e.g., gender, location, age) with dynamic behavioral patterns (e.g., discount usage, review rating, product category). 

A few assumptions made:
- `Purchase Amount` is treated as a last-transaction snapshot
- Frequency and quantity fields are assumed normalized across customers
- Subscription bias (only male subscribers) is acknowledged

---

## Analytical Questions & Rationale

### 1. Can we segment customers by shopping behavior?
Segmentation lies at the heart of **CRM (Customer Relationship Management)** and **personalized marketing**. By grouping similar customers, businesses can tailor offerings and communications to specific cohorts. Behavioral clustering also enables differentiated service strategies (e.g., VIP customers, high churn risk, discount seekers).

### 2. Can we identify high-potential promotional targets?
Efficient **resource allocation in marketing** demands precise targeting. By identifying individuals highly engaged with discounts and promotions, businesses can avoid generic campaigns and **reduce marketing waste** while increasing campaign ROI.

### 3. Can we predict subscription propensity from profile and behavior?
**Subscription models** drive predictable revenue streams. Predicting subscription likelihood helps in:
- **Retention forecasting**
- **Proactive intervention** for at-risk customers
- Designing **tiered pricing** or trial conversions

---

## Data Preparation Strategy

- **Integrity Checks**: Confirmed zero missing values.
- **Outlier Detection**: Used IQR method on `Purchase Amount` and `Age`. None were detected — suggesting a well-cleaned dataset.
- **Categorical Encoding**: Applied `LabelEncoder` for efficient model compatibility.
- **Scaling**: Standardized numerical data for KMeans and PCA using `StandardScaler`, ensuring distance calculations were meaningful and unbiased.

---

## Modeling & Analysis

### 1. KMeans Clustering (Customer Segmentation)

**Features Used**:
- `Age`, `Gender`, `Category`, `Previous Purchases`

**Justification**: These variables capture both **demographic** and **engagement-level characteristics**, serving as proxies for user intent, experience, and preferences.

**Key Insights**:
- **Optimal Clusters**: Determined via **Elbow Method** → 4 clusters
- **Cluster Validity**: Silhouette score = **0.1918** (modest separation)
- **Visual Interpretation**: PCA plot shows moderate overlap — suggesting behavioral segmentation exists but could benefit from more granularity (e.g., recency, frequency, LTV).

> Segmentation yielded coarse but actionable customer groupings. Businesses could use this to begin persona-level analysis or inform A/B test cohorts.

---

### 2. Random Forest Classification (Target Customer Detection)

**Target Definition**:
- Customers who use both **above-average discounts and promo codes** were defined as `Target Customer`

**Features Used**:
- `Age`, `Gender`, `Previous Purchases`

**Why Random Forest?**
- Handles nonlinearities and feature interactions
- Offers variable importance rankings (interpretability)
- Robust to noise and overfitting in small-to-mid datasets

**Performance**:
- **Accuracy**: 66.8%
- **Precision**: 0.63 (low false positive rate)
- **Recall**: 0.66 (moderate sensitivity)
- **F1 Score**: 0.65

> The model successfully isolates promotional users. In practice, this would allow **custom ad funnels**, **exclusive deals**, or **product bundles** to be directed at those most likely to convert.

**Reflections**:
- Features used were limited; including web engagement (e.g., session length, page views) could boost recall.
- The `Target Customer` definition is synthetic. In a real setting, this would be replaced by click-through or campaign conversion metrics.

---

### 3. Logistic Regression (Subscription Prediction)

**Features Used**:
- `Frequency of Purchases`, `Previous Purchases`, `Age`, `Gender`, `Review Rating`, `Item Purchased`, `Category`, `Location`, `Discount Applied`

**Why Logistic Regression?**
- Interpretable and well-suited for binary classification
- Models the probability of subscription with clear feature weights
- Provides a baseline for more complex models

**Performance**:
- **Accuracy**: 83.0%
- **Precision**: 0.63
- **Recall**: 1.00
- **F1 Score**: 0.77

> The model excels in **recall**, capturing all actual subscribers. While precision could be improved, the high recall is critical in **retention scenarios** where false negatives are costlier.

**Caveats**:
- All subscribers in this dataset are male — introducing gender bias.
- A probabilistic or ensemble approach might better capture partial engagement (e.g., free trial users).

---

## Tools & Libraries

- **Python** (Pandas, Scikit-learn, Matplotlib)
- **Models**: KMeans, RandomForestClassifier, LogisticRegression
- **Techniques**: LabelEncoding, StandardScaler, PCA, Elbow Method, Silhouette Analysis

---

## Key Business Takeaways

1. **Customer segments exist**, but a richer feature space (RFM, behavioral logs) could sharpen them.
2. **Target customers can be algorithmically identified** based on promotional responsiveness — enabling more profitable ad targeting.
3. **Subscription prediction** using logistic regression yields strong recall, a desirable trait when preventing customer churn.
4. **Simple models** with clean engineering offer valuable, deployable insights.

---

## Future Work

- Integrate **time-based data** for sequential modeling (e.g., churn forecasting).
- Use **web tracking** or **session logs** to enhance behavioral targeting.
- Move from logistic regression to **ensemble models** (e.g., XGBoost) for better precision.
- Introduce **model interpretability frameworks** like SHAP or LIME.
- Explore **unsupervised deep learning** for advanced embedding and representation.
