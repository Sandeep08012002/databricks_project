# Databricks notebook source
# Databricks notebook source
# MAGIC %md
# MAGIC # Customer Churn Analysis Project
# MAGIC 
# MAGIC ## Project Overview
# MAGIC This project analyzes customer churn patterns and builds a predictive model.
# MAGIC 
# MAGIC **Steps:**
# MAGIC 1. Data Ingestion & Exploration
# MAGIC 2. Data Cleaning & Preprocessing
# MAGIC 3. Feature Engineering
# MAGIC 4. Model Training (Logistic Regression & Random Forest)
# MAGIC 5. Model Evaluation
# MAGIC 6. Insights & Visualization

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Import Libraries

# COMMAND ----------

# Import Python standard libraries FIRST
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Save Python's built-in round function before importing PySpark
python_round = round

# Now import PySpark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

print("✅ Libraries imported successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Generate Synthetic Dataset

# COMMAND ----------

# Set random seed for reproducibility
random.seed(42)

# Generate 1000 customer records
synthetic_data = []

for i in range(1000):
    customer_id = "C{:04d}".format(i+1)
    gender = random.choice(["Male", "Female"])
    age = random.randint(18, 70)
    partner = random.randint(0, 1)
    
    # Generate monthly charges (avoid round conflict)
    charges = random.uniform(20, 120)
    monthly_charges = float("{:.2f}".format(charges))
    
    internet = random.randint(0, 1)
    contract = random.randint(0, 2)  # 0: Month-to-month, 1: One year, 2: Two year
    tenure = random.randint(1, 72)
    
    # Churn logic: higher probability for risky customers
    churn_prob = 0.1
    if tenure < 12:
        churn_prob += 0.3
    if monthly_charges > 80:
        churn_prob += 0.2
    if contract == 0:
        churn_prob += 0.2
    
    churn = 1 if random.random() < churn_prob else 0
    
    synthetic_data.append((customer_id, gender, age, partner, monthly_charges, internet, contract, tenure, churn))

# Define schema
schema = StructType([
    StructField("CustomerID", StringType(), True),
    StructField("Gender", StringType(), True),
    StructField("Age", IntegerType(), True),
    StructField("Partner", IntegerType(), True),
    StructField("MonthlyCharges", DoubleType(), True),
    StructField("InternetService", IntegerType(), True),
    StructField("Contract", IntegerType(), True),
    StructField("Tenure", IntegerType(), True),
    StructField("Churn", IntegerType(), True)
])

# Create DataFrame
df = spark.createDataFrame(synthetic_data, schema)

print("✅ Data loaded successfully!")
print("📊 Total Records: {}".format(df.count()))
df.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Data Exploration

# COMMAND ----------

# Display dataset information
print("Total Rows: {}".format(df.count()))
print("Total Columns: {}".format(len(df.columns)))
print("\n📊 Schema:")
df.printSchema()

# COMMAND ----------

# Summary statistics
print("📈 Statistical Summary:")
df.describe().show()

# COMMAND ----------

# Check for missing values
print("🔍 Missing Values Check:")
df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).show()

# COMMAND ----------

# Churn distribution
churn_dist = df.groupBy("Churn").count().toPandas()
print("📊 Churn Distribution:")
print(churn_dist)

churn_rate = churn_dist[churn_dist['Churn']==1]['count'].values[0] / df.count() * 100
print("\nChurn Rate: {:.2f}%".format(churn_rate))

# Visualize churn distribution
plt.figure(figsize=(8, 5))
colors = ['#2ecc71', '#e74c3c']
plt.bar(['No Churn', 'Churn'], churn_dist['count'], color=colors, edgecolor='black')
plt.xlabel('Churn Status', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold')
plt.grid(axis='y', alpha=0.3)
for i, v in enumerate(churn_dist['count']):
    plt.text(i, v + 10, str(v), ha='center', fontweight='bold')
plt.tight_layout()
display(plt.show())

# COMMAND ----------

# Analyze churn by contract type
print("📊 Churn Analysis by Contract Type:")
contract_churn = df.groupBy("Contract", "Churn").count().orderBy("Contract", "Churn")
contract_churn.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Feature Engineering

# COMMAND ----------

# Convert Gender to numeric using StringIndexer
gender_indexer = StringIndexer(inputCol="Gender", outputCol="GenderIndex")
df = gender_indexer.fit(df).transform(df)

# Create additional features
df = df.withColumn("ChargesPerMonth", F.col("MonthlyCharges") / (F.col("Tenure") + 1))
df = df.withColumn("IsNewCustomer", F.when(F.col("Tenure") < 12, 1).otherwise(0))
df = df.withColumn("IsHighSpender", F.when(F.col("MonthlyCharges") > 70, 1).otherwise(0))

# Select features for modeling
feature_cols = ["GenderIndex", "Age", "Partner", "MonthlyCharges", "InternetService", 
                "Contract", "Tenure", "ChargesPerMonth", "IsNewCustomer", "IsHighSpender"]

df_final = df.select(feature_cols + ["Churn"])

print("✅ Feature engineering completed!")
print("\n📊 Final Features: {}".format(feature_cols))
df_final.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Model Training

# COMMAND ----------

# Split data into training and test sets (80-20 split)
train_df, test_df = df_final.randomSplit([0.8, 0.2], seed=42)

train_count = train_df.count()
test_count = test_df.count()
total_count = df_final.count()

print("📚 Training set: {} rows ({:.1f}%)".format(train_count, train_count/total_count*100))
print("📝 Test set: {} rows ({:.1f}%)".format(test_count, test_count/total_count*100))

# COMMAND ----------

# Create feature vector assembler
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Feature scaling
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")

# COMMAND ----------

# Train Logistic Regression Model
print("🔄 Training Logistic Regression model...")

lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="Churn", maxIter=10)
pipeline_lr = Pipeline(stages=[assembler, scaler, lr])

model_lr = pipeline_lr.fit(train_df)

print("✅ Logistic Regression model trained successfully!")

# COMMAND ----------

# Train Random Forest Model
print("🔄 Training Random Forest model...")

rf = RandomForestClassifier(featuresCol="scaledFeatures", labelCol="Churn", numTrees=20, maxDepth=5, seed=42)
pipeline_rf = Pipeline(stages=[assembler, scaler, rf])

model_rf = pipeline_rf.fit(train_df)

print("✅ Random Forest model trained successfully!")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Model Evaluation

# COMMAND ----------

# Make predictions on test set
predictions_lr = model_lr.transform(test_df)
predictions_rf = model_rf.transform(test_df)

print("✅ Predictions generated!")

# COMMAND ----------

# Evaluate models
evaluator_auc = BinaryClassificationEvaluator(labelCol="Churn", metricName="areaUnderROC")
evaluator_acc = MulticlassClassificationEvaluator(labelCol="Churn", predictionCol="prediction", metricName="accuracy")
evaluator_precision = MulticlassClassificationEvaluator(labelCol="Churn", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="Churn", predictionCol="prediction", metricName="weightedRecall")

# Logistic Regression metrics
auc_lr = evaluator_auc.evaluate(predictions_lr)
acc_lr = evaluator_acc.evaluate(predictions_lr)
precision_lr = evaluator_precision.evaluate(predictions_lr)
recall_lr = evaluator_recall.evaluate(predictions_lr)

# Random Forest metrics
auc_rf = evaluator_auc.evaluate(predictions_rf)
acc_rf = evaluator_acc.evaluate(predictions_rf)
precision_rf = evaluator_precision.evaluate(predictions_rf)
recall_rf = evaluator_recall.evaluate(predictions_rf)

print("=" * 60)
print("📊 MODEL PERFORMANCE COMPARISON")
print("=" * 60)

print("\n🔹 Logistic Regression:")
print("   • Accuracy:  {:.4f} ({:.2f}%)".format(acc_lr, acc_lr*100))
print("   • AUC-ROC:   {:.4f}".format(auc_lr))
print("   • Precision: {:.4f}".format(precision_lr))
print("   • Recall:    {:.4f}".format(recall_lr))

print("\n🔹 Random Forest:")
print("   • Accuracy:  {:.4f} ({:.2f}%)".format(acc_rf, acc_rf*100))
print("   • AUC-ROC:   {:.4f}".format(auc_rf))
print("   • Precision: {:.4f}".format(precision_rf))
print("   • Recall:    {:.4f}".format(recall_rf))

print("\n" + "=" * 60)

if acc_rf > acc_lr:
    print("🏆 Winner: Random Forest performs better!")
else:
    print("🏆 Winner: Logistic Regression performs better!")

# COMMAND ----------

# Show sample predictions
print("🔍 Sample Predictions (Random Forest):")
predictions_rf.select("Churn", "prediction", "probability").show(15, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Feature Importance Analysis

# COMMAND ----------

# Extract feature importance from Random Forest
rf_model = model_rf.stages[-1]
feature_importance = rf_model.featureImportances.toArray()

# Create DataFrame for visualization
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("📊 Feature Importance Ranking:")
print(importance_df.to_string(index=False))

# Visualize feature importance
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(importance_df['Importance'] / importance_df['Importance'].max())
bars = plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors, edgecolor='black')
plt.xlabel('Importance Score', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
display(plt.show())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Business Insights

# COMMAND ----------

# Analyze churn patterns by contract type
print("📊 Churn Rate by Contract Type:")
contract_analysis = df.groupBy("Contract").agg(
    F.count("*").alias("Total_Customers"),
    F.sum("Churn").alias("Churned_Customers"),
    (F.sum("Churn") / F.count("*") * 100).alias("Churn_Rate_Percent")
).orderBy("Contract")

contract_analysis.show()

# COMMAND ----------

# Analyze churn by tenure groups
df_with_tenure_group = df.withColumn(
    "TenureGroup",
    F.when(F.col("Tenure") < 12, "0-12 months")
    .when(F.col("Tenure") < 24, "12-24 months")
    .when(F.col("Tenure") < 48, "24-48 months")
    .otherwise("48+ months")
)

print("📊 Churn Rate by Tenure Group:")
tenure_analysis = df_with_tenure_group.groupBy("TenureGroup").agg(
    F.count("*").alias("Total_Customers"),
    F.sum("Churn").alias("Churned_Customers"),
    (F.sum("Churn") / F.count("*") * 100).alias("Churn_Rate_Percent")
).orderBy("TenureGroup")

tenure_analysis.show()

# COMMAND ----------

# Analyze churn by monthly charges
df_with_charge_group = df.withColumn(
    "ChargeGroup",
    F.when(F.col("MonthlyCharges") < 40, "Low ($20-40)")
    .when(F.col("MonthlyCharges") < 70, "Medium ($40-70)")
    .otherwise("High ($70+)")
)

print("📊 Churn Rate by Monthly Charges:")
charge_analysis = df_with_charge_group.groupBy("ChargeGroup").agg(
    F.count("*").alias("Total_Customers"),
    F.sum("Churn").alias("Churned_Customers"),
    (F.sum("Churn") / F.count("*") * 100).alias("Churn_Rate_Percent")
).orderBy("ChargeGroup")

charge_analysis.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 🎯 Key Findings & Recommendations

# COMMAND ----------

print("=" * 70)
print("🎯 KEY FINDINGS & BUSINESS RECOMMENDATIONS")
print("=" * 70)

print("\n📈 MODEL PERFORMANCE:")
print("   • Best Model: Random Forest with {:.2f}% accuracy".format(acc_rf*100))
print("   • Can predict churn with {:.2f}% reliability (AUC)".format(auc_rf*100))

print("\n🔍 TOP CHURN PREDICTORS:")
top_3_features = importance_df.head(3)
for idx, row in top_3_features.iterrows():
    print("   {}. {} (Importance: {:.4f})".format(idx+1, row['Feature'], row['Importance']))

print("\n💡 BUSINESS RECOMMENDATIONS:")
print("   1. Focus retention efforts on customers in first 12 months")
print("   2. Offer incentives to convert month-to-month to longer contracts")
print("   3. Review pricing strategy for high-charge customers")
print("   4. Implement early warning system using this model")
print("   5. Target high-risk customers with personalized retention campaigns")

print("\n" + "=" * 70)

