# Databricks notebook source
import os

os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/ecommerce/mlflow_tmp"

# COMMAND ----------

model_df = spark.read.table("workspace.ecommerce.ml_training_dataset")

# COMMAND ----------

train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(
    labelCol="label",
    metricName="areaUnderROC"
)

# COMMAND ----------

import mlflow

mlflow.set_experiment("/Users/aishwaryaprabha29@gmail.com/ecommerce_mlflow_tracking")

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.classification import LogisticRegression

with mlflow.start_run(run_name="logistic_regression_baseline"):

    lr = LogisticRegression(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight"
    )

    model = lr.fit(train_df)
    preds = model.transform(test_df)

    auc = evaluator.evaluate(preds)

    # Log params
    mlflow.log_param("model_type", "LogisticRegression")

    # Log metric
    mlflow.log_metric("AUC", auc)

    # Log model
    mlflow.spark.log_model(model, "logistic_regression_model")

    print("Logistic Regression AUC:", auc)

# COMMAND ----------

with mlflow.start_run(run_name="random_forest_default"):

    from pyspark.ml.classification import RandomForestClassifier

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        numTrees=100,
        maxDepth=8,
        seed=42
    )

    model = rf.fit(train_df)
    preds = model.transform(test_df)

    auc = evaluator.evaluate(preds)

    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("numTrees", 100)
    mlflow.log_param("maxDepth", 8)

    # Log metric
    mlflow.log_metric("AUC", auc)

    # Log model
    mlflow.spark.log_model(model, "random_forest_model")

    print("Random Forest AUC:", auc)

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
import mlflow

trees_list = [100, 200]
depth_list = [8, 10]

for trees in trees_list:
    for depth in depth_list:

        with mlflow.start_run(run_name=f"rf_trees_{trees}_depth_{depth}"):

            rf = RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                weightCol="classWeight",
                numTrees=trees,
                maxDepth=depth,
                seed=42
            )

            model = rf.fit(train_df)
            preds = model.transform(test_df)

            auc = evaluator.evaluate(preds)

            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("numTrees", trees)
            mlflow.log_param("maxDepth", depth)
            mlflow.log_metric("AUC", auc)

            print(f"Trees: {trees}, Depth: {depth}, AUC: {auc}")

# COMMAND ----------

import mlflow
from pyspark.ml.classification import RandomForestClassifier

with mlflow.start_run(run_name="rf_final_selected_model"):

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        numTrees=100,
        maxDepth=8,
        seed=42
    )

    final_model = rf.fit(train_df)
    preds = final_model.transform(test_df)

    final_auc = evaluator.evaluate(preds)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("numTrees", 100)
    mlflow.log_param("maxDepth", 8)
    mlflow.log_metric("AUC", final_auc)

    mlflow.spark.log_model(final_model, "final_random_forest_model")

    print("Final Model AUC:", final_auc)

# COMMAND ----------

from mlflow.models.signature import infer_signature

# Take small sample from test data
sample_input = test_df.limit(10)

# Get predictions
sample_output = final_model.transform(sample_input)

# Infer signature
signature = infer_signature(
    sample_input.toPandas(),
    sample_output.select("prediction").toPandas()
)

# COMMAND ----------

with mlflow.start_run(run_name="rf_final_selected_model_signed"):

    rf = RandomForestClassifier(
        featuresCol="features",
        labelCol="label",
        weightCol="classWeight",
        numTrees=100,
        maxDepth=8,
        seed=42
    )

    final_model = rf.fit(train_df)
    preds = final_model.transform(test_df)

    final_auc = evaluator.evaluate(preds)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("numTrees", 100)
    mlflow.log_param("maxDepth", 8)
    mlflow.log_metric("AUC", final_auc)

    # Infer signature
    sample_input = test_df.limit(10)
    sample_output = final_model.transform(sample_input)

    signature = infer_signature(
        sample_input.toPandas(),
        sample_output.select("prediction").toPandas()
    )

    mlflow.spark.log_model(
        final_model,
        "final_random_forest_model",
        signature=signature
    )

    print("Final Model AUC:", final_auc)

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "workspace.ecommerce.rf_purchase_prediction"

# Get the latest run ID
run_id = mlflow.last_active_run().info.run_id

model_uri = f"runs:/{run_id}/final_random_forest_model"

registered_model = mlflow.register_model(
    model_uri=model_uri,
    name=model_name
)

print("Model registered successfully.")

# COMMAND ----------

client.transition_model_version_stage(
    name=model_name,
    version=registered_model.version,
    stage="Staging"
)

# COMMAND ----------

