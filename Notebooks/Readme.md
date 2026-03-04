# Notebooks

This folder contains the core notebooks/scripts used to build the **End-to-End Ecommerce ML Pipeline**.

## Files

**Bronze_Layer.py**  
Ingests raw ecommerce CSV event data and stores it in the Bronze Delta table.

**Silver_Layer.py**  
Transforms raw events into user-level behavioral features used for machine learning.

**mlflow_tracking.py**  
Trains purchase prediction models (Logistic Regression, Random Forest) and tracks experiments using MLflow.

**gold_user_prediction.py**  
Runs batch inference using the registered model and stores predictions in Gold tables.

**AI_recommendation_system.py**  
Builds a product recommendation system using ALS collaborative filtering.

## Pipeline Flow
Bronze → Silver → ML Training → Model Registry → Batch Inference → Gold Predictions

**This pipeline predicts **user purchase probability** and generates **personalized product recommendations**.**
