# Databricks notebook source
dbutils.widgets.text("run_date", "")
run_date = dbutils.widgets.get("run_date")

if run_date == "":
    raise ValueError("run_date parameter is required")

# COMMAND ----------

df = spark.read.format("csv") \
    .option("header", "True") \
    .option("inferSchema", "True") \
    .load("/Volumes/workspace/ecommerce/ecommerce_data/*.csv")


# COMMAND ----------

from pyspark.sql.functions import to_date, col

df = df.withColumn("event_date", to_date(col("event_time")))

incremental_df = df.filter(col("event_date") == run_date)



# COMMAND ----------

print("Rows to be written:", incremental_df.count())


# COMMAND ----------

incremental_df.write \
  .format("delta") \
  .mode("append") \
  .saveAsTable("workspace.ecommerce.bronze_ecommerce_events")



# COMMAND ----------

spark.table("workspace.ecommerce.bronze_ecommerce_events").count()


# COMMAND ----------

spark.table("workspace.ecommerce.bronze_ecommerce_events") \
     .groupBy("event_date") \
     .count() \
     .orderBy("event_date") \
     .show(50, False)


# COMMAND ----------

