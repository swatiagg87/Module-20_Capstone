**Swati Aggarwal**

#### Project Title

Predicting Customer Churn in E-Commerce Using RFM Segmentation and Machine Learning


#### Executive Summary

This project aims to predict which e-commerce customers are at risk of churning and provide actionable insights to improve retention. Using the Brazilian E-Commerce Public Dataset (Olist), we performed data cleaning, exploratory data analysis (EDA), RFM segmentation, and churn prediction modeling. Results show that churn is highly predictable with simple behavioral features, and that a small group of loyal customers contributes most revenue. Logistic Regression and ensemble models achieved strong performance (ROC-AUC ~0.96–0.98), making churn detection both reliable and business-ready.

#### Rationale

Customer retention is more cost-effective than customer acquisition — studies suggest it costs 5–7x more to acquire a new customer than to retain an existing one. For e-commerce, churn directly impacts profitability and growth. By predicting churn risk early, businesses can proactively engage customers, tailor retention campaigns, and allocate marketing resources efficiently.

#### Research Question

Which customers are at risk of churning?

How can we segment customers into meaningful groups to drive retention strategies?

What behavioral factors (Recency, Frequency, Monetary spend) best predict churn?

#### Data Sources

Brazilian E-Commerce Public Dataset (Olist, Kaggle)

Orders, Customers, Payments, Products, and Reviews data from 2016–2018

~119K orders, ~92K unique customers

Cleaned and preprocessed to remove missing values, duplicates, and outliers.

#### Methodology

Data Cleaning & Feature Engineering: Handled missing/null values, removed duplicates, treated outliers. Engineered Recency, Frequency, Monetary (RFM), tenure, and average order value.

EDA: Visualized purchase patterns, revenue concentration, customer geographies, and recency distributions.

RFM Segmentation: Clustered customers into Champions, Loyal, Potential Loyalists, and At Risk segments. Created RFM heatmaps to visualize segment spending.

Modeling:

Logistic Regression (baseline)

Random Forest, Gradient Boosting (ensemble methods)

Evaluation Metrics: Accuracy, Precision, Recall, F1, ROC-AUC.

Validation: Stratified 5-fold cross-validation and Lift/Gain charts for business relevance.

#### Results

EDA: Most customers are one-time buyers. ~40% of customers generate ~80% of revenue. Customers concentrated in São Paulo and Rio de Janeiro.

RFM Segmentation: Clear groups identified — Champions (high value, engaged), Loyal (steady base), Potential Loyalists (promising), At Risk (inactive, low spend).

Modeling:

Logistic Regression achieved AUC ~0.97, with high recall (92%) and precision (99%).

Random Forest and Gradient Boosting performed similarly, with GBM slightly higher (AUC ~0.98).

Lift/Gain chart: top 20% of flagged customers captured ~70% of actual churners → retention efforts can be highly efficient.

Cross-validation: Stable results across folds, proving models generalize well.

#### Next Steps

Refine churn definition: Use fixed windows (e.g., 90 days inactivity) instead of mean recency.

Feature expansion: Add product categories, reviews, seasonal patterns, and customer service interactions.

Actionable deployment: Automate weekly churn scoring in CRM, triggering targeted campaigns.

Experimentation: Run A/B tests on retention interventions (discounts, emails, personalized offers).

Monitoring & retraining: Update models quarterly, track drift, and adjust thresholds to balance precision vs. recall.

#### Outline of Project

Link to Notebook 1 – Retail Churn Model


Contact and Further Information

For questions or further information, please contact:
Swati Aggarwal
Swati.aggarwal.ait09@gmail.com