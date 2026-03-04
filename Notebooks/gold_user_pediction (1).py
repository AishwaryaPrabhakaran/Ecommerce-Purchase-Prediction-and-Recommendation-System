# Databricks notebook source
import os
os.environ["MLFLOW_DFS_TMP"] = "/Volumes/workspace/ecommerce/mlflow_tmp"

# COMMAND ----------

import mlflow.spark

model_uri = "models:/workspace.ecommerce.rf_purchase_prediction/1"

loaded_model = mlflow.spark.load_model(model_uri)

# COMMAND ----------

scoring_df = spark.read.table("workspace.ecommerce.ml_training_dataset")

# COMMAND ----------

scoring_df = scoring_df.drop("label", "classWeight")

# COMMAND ----------

predictions = loaded_model.transform(scoring_df)

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

predictions_clean = predictions.withColumn(
    "purchase_probability",
    vector_to_array(col("probability"))[1]
)

# COMMAND ----------

predictions_clean.select(
    "user_id",
    "purchase_probability",
    "prediction"
).show(5)

# COMMAND ----------

from pyspark.sql.functions import lit, current_timestamp

model_version = "1"

predictions_gold = predictions_clean.select(
    "user_id",
    "purchase_probability",
    "prediction"
).withColumn(
    "model_version", lit(model_version)
).withColumn(
    "scored_at", current_timestamp()
)

# COMMAND ----------

predictions_gold.printSchema()

# COMMAND ----------

predictions_gold.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable("workspace.ecommerce.gold_user_predictions_history")

# COMMAND ----------

spark.read.table("workspace.ecommerce.gold_user_predictions_history").count()

# COMMAND ----------

predictions_gold.write \
    .format("delta") \
    .mode("overwrite") \
    .saveAsTable("workspace.ecommerce.gold_user_predictions_snapshot")

# COMMAND ----------

from delta.tables import DeltaTable

snapshot_table = DeltaTable.forName(
    spark,
    "workspace.ecommerce.gold_user_predictions_snapshot"
)

snapshot_table.alias("target").merge(
    predictions_gold.alias("source"),
    "target.user_id = source.user_id"
).whenMatchedUpdate(set={
    "purchase_probability": "source.purchase_probability",
    "prediction": "source.prediction",
    "model_version": "source.model_version",
    "scored_at": "source.scored_at"
}).whenNotMatchedInsertAll().execute()

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import percent_rank

window_spec = Window.orderBy(predictions_gold.purchase_probability.desc())

ranked_df = predictions_gold.withColumn(
    "percent_rank",
    percent_rank().over(window_spec)
)

top_buyers = ranked_df.filter("percent_rank <= 0.10")

top_buyers.select("user_id", "purchase_probability").show(10)

# COMMAND ----------

