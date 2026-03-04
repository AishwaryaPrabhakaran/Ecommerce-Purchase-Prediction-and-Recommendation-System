# Databricks notebook source
dbutils.widgets.text("run_date", "")
run_date = dbutils.widgets.get("run_date")

if run_date == "":
    raise ValueError("run_date parameter is required")


# COMMAND ----------

bronze_df = spark.read.table("workspace.ecommerce.bronze_ecommerce_events")


# COMMAND ----------

from pyspark.sql.functions import to_date, col
# Ensure event_date column exists (safety step)
bronze_df = bronze_df.withColumn("event_date", to_date(col("event_time")))

# COMMAND ----------

# Filter only today's data
daily_df = bronze_df.filter(col("event_date") == run_date)


# COMMAND ----------

print("Daily records:", daily_df.count())

# COMMAND ----------

# Identify users who had activity on run_date
affected_users = daily_df.select("user_id").distinct()

print("Affected users:", affected_users.count())

# COMMAND ----------

# Get full historical data for affected users
full_user_df = bronze_df.join(affected_users, on="user_id", how="inner")

print("Full history records for affected users:", full_user_df.count())

# COMMAND ----------

from pyspark.sql.functions import (
    count, sum, when, max, countDistinct, coalesce, lit
)

updated_user_features = full_user_df.groupBy("user_id").agg(
    count("*").alias("total_events"),
    countDistinct("user_session").alias("sessions"),
    count(when(col("event_type") == "purchase", True)).alias("total_purchases"),
    count(when(col("event_type") == "cart", True)).alias("total_cart_adds"),
    countDistinct("product_id").alias("unique_products"),
    max("price").alias("max_viewed_price"),
    coalesce(
        max(when(col("event_type") == "purchase", col("price"))),
        lit(0)
    ).alias("max_purchase_price"),
    max("event_time").alias("last_event_time"),
    coalesce(
        sum(when(col("event_type") == "purchase", col("price"))),
        lit(0)
    ).alias("total_spent")
)

print("Recomputed users:", updated_user_features.count())


# COMMAND ----------

from delta.tables import DeltaTable

silver_table = DeltaTable.forName(
    spark, 
    "workspace.ecommerce.silver_user_features"
)

silver_table.alias("target").merge(
    updated_user_features.alias("source"),
    "target.user_id = source.user_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# COMMAND ----------

spark.table("workspace.ecommerce.silver_user_features").count()


# COMMAND ----------

spark.table("workspace.ecommerce.silver_user_features") \
    .filter(col("last_event_time").cast("date") == "2019-11-21") \
    .count()
