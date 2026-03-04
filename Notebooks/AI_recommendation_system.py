# Databricks notebook source
spark.read.table("workspace.ecommerce.bronze_streaming_events") \
    .select("event_type") \
    .distinct() \
    .show()

# COMMAND ----------

# DBTITLE 1,Untitled
from pyspark.sql.functions import col

bronze_df = spark.read.table("workspace.ecommerce.bronze_streaming_events")

interaction_df = bronze_df.select(
    col("user_id"),
    col("product_id"),
    col("event_type")
)

interaction_df.show(5)

# COMMAND ----------

from pyspark.sql.functions import when

rating_df = interaction_df.withColumn(
    "rating",
    when(col("event_type") == "view", 1.0)
    .when(col("event_type") == "cart", 3.0)
    .when(col("event_type") == "purchase", 5.0)
)

rating_df.show(5)

# COMMAND ----------

rating_df = rating_df.select(
    "user_id",
    "product_id",
    "rating"
)

# COMMAND ----------

from pyspark.sql.functions import max as spark_max

final_interaction_df = rating_df.groupBy(
    "user_id",
    "product_id"
).agg(
    spark_max("rating").alias("rating")
)

final_interaction_df.show(5)

# COMMAND ----------

final_interaction_df.printSchema()
final_interaction_df.count()
final_interaction_df.select("rating").distinct().show()

# COMMAND ----------

from pyspark.sql.functions import col

final_interaction_df = final_interaction_df \
    .withColumn("user_id", col("user_id").cast("int")) \
    .withColumn("product_id", col("product_id").cast("int"))

# COMMAND ----------

final_interaction_df.printSchema()

# COMMAND ----------

final_interaction_df.count()

# COMMAND ----------

final_interaction_df.select("user_id").distinct().count()

# COMMAND ----------

final_interaction_df.select("product_id").distinct().count()

# COMMAND ----------

from pyspark.ml.recommendation import ALS

als = ALS(
    userCol="user_id",
    itemCol="product_id",
    ratingCol="rating",
    rank=10,
    maxIter=5,
    regParam=0.1,
    coldStartStrategy="drop",
    implicitPrefs=False
)

als_model = als.fit(final_interaction_df)

# COMMAND ----------

user_recs = als_model.recommendForAllUsers(5)

# COMMAND ----------

user_recs.show(5, truncate=False)

# COMMAND ----------

sample_users = final_interaction_df \
    .select("user_id") \
    .distinct() \
    .limit(1000)

# COMMAND ----------

predictions = als_model.transform(
    sample_users.crossJoin(
        final_interaction_df.select("product_id").distinct()
    )
)

# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col

window_spec = Window.partitionBy("user_id") \
    .orderBy(col("prediction").desc())

ranked = predictions.withColumn(
    "rank",
    row_number().over(window_spec)
)

top5 = ranked.filter(col("rank") <= 5)

top5.select("user_id", "product_id", "prediction").show(10)

# COMMAND ----------

users_df = final_interaction_df.select("user_id").distinct()

# COMMAND ----------

users_df = users_df.sample(0.1)

# COMMAND ----------

user_recs = als_model.recommendForUserSubset(users_df, 5)

# COMMAND ----------

from pyspark.sql.functions import explode, col

exploded_recs = user_recs \
    .withColumn("rec", explode(col("recommendations"))) \
    .select(
        col("user_id"),
        col("rec.product_id").alias("recommended_product"),
        col("rec.rating").alias("predicted_score")
    )

exploded_recs.show(5)

# COMMAND ----------

