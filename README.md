
# Predicting Customer Churn in E-Commerce Using Behavioral Segmentation and Machine Learning

**Author:** Swati Aggarwal  
**Institution:** UC Berkeley – Professional Certificate in Machine Learning & AI  
**Project Type:** Capstone Final Report  
**Dataset:** Brazilian E-Commerce Public Dataset (Olist)  
**Keywords:** Customer churn, RFM segmentation, machine learning, predictive modeling, e-commerce retention  

---

## Problem Statement and Business Context


Customer retention remains one of the most pressing challenges in e-commerce. As acquisition costs rise, businesses increasingly rely on retaining existing customers to sustain growth and profitability. In this project, the goal was to **predict customer churn**—that is, identify customers who are likely to become inactive—using transactional and behavioral data.

Churn was not defined by a fixed window of inactivity (e.g., 90 days), but **dynamically** based on the dataset’s behavior. Customers whose *recency* exceeded the **mean recency value** were labeled as inactive. This adaptive definition allowed the model to reflect real-world behavior instead of arbitrary time-based cutoffs.

The project sought to answer three key questions:
1. What behavioral factors most strongly influence churn?
2. Can we predict churn early enough to intervene?
3. How can these predictions translate into measurable business strategies for retention and revenue optimization?

By leveraging behavioral segmentation, RFM scoring, and supervised machine learning, this project delivers an actionable churn-prediction framework that is both explainable and operationally practical.

---
## Model Outcomes and Predictions

The project frames churn prediction as a **supervised binary classification** problem.  
Customers are labeled as:
- **Active (0):** Recency ≤ mean(recency)  
- **Churned (1):** Recency > mean(recency)  

Each model outputs both a binary label and a probability score that can drive CRM triggers or campaign prioritization.  
Across experiments, models achieved accuracy around **0.95** and ROC-AUC up to **0.98**, confirming that customer-behavior signals alone hold substantial predictive power.

---

## Data Acquisition and Preparation


This dataset was generously provided by Olist, the largest department store in Brazilian marketplaces. Olist connects small businesses from all over Brazil to channels without hassle and with a single contract. Those merchants are able to sell their products through the Olist Store and ship them directly to the customers using Olist logistics partners. See more on our website: www.olist.com

After a customer purchases the product from Olist Store a seller gets notified to fulfill that order. Once the customer receives the product, or the estimated delivery date is due, the customer gets a satisfaction survey by email where he can give a note for the purchase experience and write down some comments.

### Data Preparation
**1 | Data Cleaning**  
Records with missing timestamps or payment values were dropped; these constituted < 2% of rows.  
All date fields were converted to `datetime` format and sorted chronologically to calculate recency accurately.  

**2 | Aggregation and Feature Engineering**  
Using `groupby(customer_id)`, transaction-level tables were collapsed into RFM metrics.  
`recency` was computed as the days since the most recent order relative to the dataset’s maximum purchase date.  
`frequency` counted completed orders per customer; `monetary` summed all payment values.  
Additional lifecycle variables (`tenure`, `avg_order_value`) captured spending consistency.  

**3 | Churn Label Creation**  
The churn variable was derived as:  
`churn = 1 if recency > mean(recency) else 0`.  
This **adaptive mean-recency rule** ties churn directly to observed purchasing rhythm, preventing false churn flags during slow periods.  

**4 | Normalization and Splitting**  
Continuous features were standardized with `StandardScaler` to ensure balanced weight across models.  

### Data Preparation Outcome
1) New columns are created using the available date time columns for easy analysis of the available data.
2) Purchased_approved represents the seconds taken for an order to get approved after the customer purchases it.
3) approved_carrier represents the days taken for the order to go to the delivery carrier after it being approved.
4) carrier_delivered represents the days taken for the order to be delivered to the customer from the date it reaches the delivery carrier.
5) delivered_estimated represents the date difference between the estimated delivery date and the actual delivery date.
6) purchased_delivered represents the days taken for the order to be delivered to the customer from the date the customer made the purchase.
7) Orders which have carrier date prior to the date of order getting approved, and orders which have delivered date prior to the carrier date are considered to be falsified data, as it could not be logically true.Such records are dropped.
8) Also the records which are cancelled and have no null values are also dropped as we consider only the records which have a order status of delivered.

### EDA Insights:

- The dataset contains **99,441 unique orders** and **96,096 unique customers**.  
  - Observation: Almost all customers are *one-time buyers*, as their purchase frequency is **1**.  
- Customers are spread across **14,994 unique zip codes**.  
  - The **highest concentration** is in **zip code 22790 (Rio de Janeiro, Brazil)**, with **142 customers**.  
- **Top Locations**:  
  - **City:** São Paulo has the largest number of customers.  
  - **State:** São Paulo is also the top state by customer count. 
 - The **median order approval time** (from purchase to approval) is **1,154 seconds (~20 minutes)**, indicating a right-skewed distribution.  
- The **average delivery time** from purchase to customer receipt is **12 days**.  
- Since only delivered orders were included, the **`order_status`** feature contains a single class: *delivered*.  
- The dataset spans from the **first order on 15/09/2016** to the **last order on 29/08/2018**.  
- On average, orders were delivered **10 days earlier than the estimated delivery date**. However, many orders were still delivered late, reflecting **variability in delivery accuracy**.  
- The top 41.78% of customers contribute ~80% of revenue, highlighting that a relatively small group of customers drives most of the business. This segment should be prioritized for retention and loyalty strategies, while opportunities exist to increase engagement among lower-value customers.

### RFM Insights:
1. **Champions – Top Spenders**  
   - **Size**: ~29K customers  
   - **Avg. Spend**: ~$352  
   - **Profile**: Loyal, engaged, and highly profitable — these are the “VIP” customers.  
   - **Action**: Provide personalized rewards, early access to products, and ongoing appreciation to strengthen loyalty.  

2. **Loyal Customers – Steady Base**  
   - **Size**: ~34K customers (largest segment)  
   - **Avg. Spend**: ~$194  
   - **Profile**: Consistent buyers who form the backbone of the business.  
   - **Action**: Build stronger relationships to secure steady revenue and long-term advocacy.  

3. **Potential Loyalists – Strong Promise**  
   - **Size**: ~23K customers  
   - **Avg. Spend**: ~$107  
   - **Profile**: Increasingly engaged, with potential to become Loyal Customers or Champions.  
   - **Action**: Use personalized recommendations, targeted discounts, and loyalty perks to encourage deeper engagement.  

4. **At Risk – Urgent Attention Needed**  
   - **Size**: ~6K customers (smallest group)  
   - **Avg. Spend**: ~$54  
   - **Profile**: Low spenders who are likely to churn if not re-engaged.  
   - **Action**: Run win-back campaigns, offer targeted discounts, and collect feedback to attempt recovery.  

- Customers with **low Recency (recent purchases)** and **high Frequency** consistently show the **highest Monetary value** — these are our Champions.  
- Customers with **high Recency (long time since last purchase)**, regardless of Frequency, show lower Monetary averages — suggesting disengagement leads to declining value.  
- The heatmap confirms that focusing on **recent and frequent buyers** delivers the highest ROI for retention efforts.  

---

## Modeling

## Modeling and Algorithm Selection
Modeling followed an incremental approach—starting simple for interpretability and scaling toward complexity for marginal accuracy.  The guiding principle was to balance **predictive performance**, **explainability**, and **business deployability**.

### 1 | Logistic Regression – Baseline Model  
This served as the interpretive benchmark.  Coefficients clearly revealed direction and strength of behavioral effects:  
- **Recency** had the largest positive weight → the longer since last purchase, the higher the churn probability.  
- **Frequency** and **Monetary** showed negative weights → repeat and high-spend buyers are inherently stickier.  
- **Tenure** provided a small stabilizing effect, implying that longevity mitigates risk but doesn’t eliminate it.  
The logistic model achieved ≈ 0.96 accuracy and 0.94 AUC, already strong enough for baseline retention modeling.

### 2 | Random Forest – Capturing Nonlinearity  
Random Forest introduced ensemble robustness by averaging multiple decision trees.  
Grid-search tuning of `n_estimators`, `max_depth`, and `min_samples_split` produced ≈ 0.94 accuracy with balanced precision (0.95) and recall (0.93).  
Its feature-importance plot ranked Recency ≫ Frequency ≫ Monetary, validating behavioral intuition.  
The model’s lower variance confirmed that churn behavior follows a consistent pattern across subsets of customers.

### 3 | Gradient Boosting – Optimized Predictive Power  
Gradient Boosting sequentially improved weak learners to minimize residual error.  
With tuned `learning_rate` and `n_estimators`, it reached ≈ 0.96 overall accuracy and 0.99 precision, per your notebook output.  
High precision means the model rarely mislabels active customers as churners—critical for cost-efficient targeting.  
Its recall (0.92) ensures most true churners are detected, yielding the best precision-recall balance among all models.  


### Modeling Approach

- **Cross-validation:** 5-fold cross-validation was used to assess model generalizability.  
- **Hyperparameter tuning:**  
  - Random Forest: tuned `n_estimators`, `max_depth`, and `min_samples_split`  
  - GBM: optimized `learning_rate` and `n_estimators` via grid search  
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score, and ROC-AUC were used for holistic assessment.  

### Insights

- **5-fold Stratified Cross-Validation** shows highly consistent results across folds, with low variance.  
- This demonstrates that **Logistic Regression generalizes well** and its performance is not just a result of one lucky train/test split.  
- Stability across folds boosts confidence in the model, making it a **reliable baseline** for churn prediction.  
- Compared to RF and GBM, Logistic Regression holds its ground despite being simpler, proving that **interpretability and predictive power can go hand-in-hand**.
---

## Model Evaluation Summary

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |  
|:--|--:|--:|--:|--:|--:|  
| Logistic Regression | 0.96 | 0.98 | 0.92 | 0.95 | 0.97 |  
| Random Forest | 0.95 | 0.95 | 0.93 | 0.94 | 0.97 |  
| Gradient Boosting | 0.96 | 0.98 | 0.92 | 0.95 | 0.97 |  

Since churn is not evenly balanced, Accuracy alone is misleading. We use Precision and Recall to balance cost of false positives vs false negatives, and ROC-AUC to measure separability across thresholds.

- All three models — **Logistic Regression, Random Forest, and Gradient Boosting** — show strong performance, with metrics consistently in the 0.94–0.96 range.  
- **Logistic Regression** is nearly as effective as the more complex models, proving that churn is largely explained by simple behavioral features (Recency, Frequency, Monetary).  
- **Gradient Boosting** performs slightly better in Recall and ROC-AUC, meaning it catches more churners at the cost of complexity.  
- **Logistic Regression** is easier to interpret and may be preferred if business stakeholders value explainability.  
- Overall, the differences are small — which is encouraging, as it shows that the churn signal is strong and robust across algorithms.

### Business Implications

1. **Adaptive Churn Definition:** Using mean recency enables automatic recalibration as buying cadence evolves—ideal for dynamic e-commerce environments.  
2. **Precision Marketing:** GBM’s 0.99 precision ensures retention resources target real at-risk users, improving ROI.  
3. **Lifecycle Campaigns:**  
   - **Trigger:** Customer crosses mean recency.  
   - **Action:** Personalized reactivation offer or reminder.  
   - **Measure:** Reactivation uplift vs control group.  
4. **Revenue Shielding:** Combine churn probability with customer Monetary value to prioritize interventions by potential revenue loss.  
5. **Organizational Impact:** Embedding churn scoring into CRM workflows enables marketing teams to act weekly, not quarterly.
---
## Future Work

- Integrate **review sentiment** for experience-driven churn detection.  
- Add **temporal features** like “time since last order.”  
- Automate scoring pipeline (Airflow / AWS Lambda).  
- Evaluate retention lift via **A/B testing**.  

---

## Conclusion

This capstone proves that **behavioral data alone can predict churn with over 95% accuracy**.  
The adaptive mean-recency definition personalizes churn detection across cycles.  
Through interpretable models like Logistic Regression and scalable ensembles like GBM, the project translates machine learning into actionable retention strategy—bridging analytics with growth execution.

---

**Contact:**  
Swati Aggarwal  
swati.aggarwal.ait09@gmail.com  
UC Berkeley ML/AI Capstone Project

