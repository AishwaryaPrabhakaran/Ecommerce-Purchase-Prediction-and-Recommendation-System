# End-to-End Pipeline Flow Documentation

This project implements a **production-style machine learning pipeline for an ecommerce platform**, built using **Databricks, Delta Lake, Apache Spark, and MLflow**.  
The system predicts **user purchase probability** and also generates **personalized product recommendations**.

---
## Architecture Diagram
![Architecture Diagram](architecture_diagram.png)

---

# Step 1 — Data Ingestion (Bronze Layer)

Raw ecommerce event data is ingested from CSV files into the **Bronze Delta table**.

This layer stores **raw, unprocessed data exactly as received from the source systems**.

Typical events include:

- Product views
- Cart additions
- Purchases
- Session interactions

The Bronze layer acts as a **reliable landing zone for data** and ensures that the original data is preserved for **auditing and reprocessing**.

**Output table**
workspace.ecommerce.bronze_ecommerce_events

---

# Step 2 — Data Processing & Feature Engineering (Silver Layer)

The Silver layer transforms **raw event logs into structured user-level behavioral features**.

### Examples of features created

- Total user events
- Number of sessions
- Number of purchases
- Unique products viewed
- Maximum viewed price
- Total amount spent

To improve performance, the pipeline **recomputes features only for users who had activity on the current run date**.

This **incremental approach** reduces compute cost and allows the system to **scale efficiently**.

**Output table**
workspace.ecommerce.silver_user_features

---

# Step 3 — Machine Learning Training Dataset Creation

The engineered features from the **Silver layer** are used to build a **machine learning training dataset**.

The dataset contains:

- User behavioral features
- Engineered feature vectors
- Label column indicating whether the user made a purchase

This dataset is used to train **classification models that predict purchase likelihood for each user**.

**Training dataset**
workspace.ecommerce.ml_training_dataset

---

# Step 4 — Model Training and Experiment Tracking

Multiple machine learning models are trained using the training dataset.

Models experimented in this project include:

- Logistic Regression
- Random Forest

Model performance is evaluated using the **Area Under ROC Curve (AUC)** metric.

All experiments, parameters, and evaluation metrics are logged using **MLflow Tracking**, enabling:

- Experiment reproducibility
- Easy comparison between models
- Transparent ML experimentation

---

# Step 5 — Model Registration and Versioning

The best-performing model is registered in the **MLflow Model Registry**.

This enables:

- Model version control
- Controlled deployment workflows
- Complete model lifecycle management

The registered model is stored as:
workspace.ecommerce.rf_purchase_prediction

Models can be promoted across stages such as:
Staging → Production

---

# Step 6 — Batch Inference Pipeline

The registered model is loaded from the **MLflow Model Registry** and used to perform **batch predictions** on user feature data.

For each user, the model generates:

- Purchase probability
- Predicted purchase label

This step allows the system to **identify users most likely to purchase in the near future**.

---

# Step 7 — Gold Layer (Business-Ready Predictions)

Model predictions are stored in **Gold tables**, which are optimized for **analytics and business consumption**.

Two types of prediction tables are maintained.

### Prediction History Table

Stores all historical predictions with model version and scoring timestamp.

---

# Step 6 — Batch Inference Pipeline

The registered model is loaded from the **MLflow Model Registry** and used to perform **batch predictions** on user feature data.

For each user, the model generates:

- Purchase probability
- Predicted purchase label

This step allows the system to **identify users most likely to purchase in the near future**.

---

# Step 7 — Gold Layer (Business-Ready Predictions)

Model predictions are stored in **Gold tables**, which are optimized for **analytics and business consumption**.

Two types of prediction tables are maintained.

### Prediction History Table

Stores all historical predictions with model version and scoring timestamp.
workspace.ecommerce.gold_user_predictions_history

### Prediction Snapshot Table

Maintains the **latest prediction for each user**.
workspace.ecommerce.gold_user_predictions_snapshot

This allows business teams to easily **query the latest high-probability buyers**.

---

# Step 8 — Business Applications

The Gold layer predictions can be consumed by various downstream systems, including:

- Marketing targeting systems
- Personalized promotions
- CRM campaigns
- BI dashboards
- Customer segmentation analysis

Example use case:

Marketing teams can identify the **top 10% high-purchase-probability users** and target them with **personalized offers**.

---

# Parallel Pipeline — Product Recommendation System

In addition to purchase prediction, the system also includes a **recommendation engine**.

User interaction events (**view, cart, purchase**) are converted into **interaction ratings**, and an **ALS collaborative filtering model** is trained.

The model generates **Top-N product recommendations for each user**, enabling personalized shopping experiences similar to recommendation systems used by major ecommerce platforms.

---

# Key Takeaways

This architecture demonstrates a **complete production-style ML system**, including:

- Medallion data architecture (**Bronze → Silver → Gold**)
- Incremental feature engineering
- ML experiment tracking using **MLflow**
- Model versioning and registry
- Batch inference pipelines
- Business-ready prediction tables
- Product recommendation system integration

---
