import React, { useState, useEffect, useRef } from 'react';

const questions = [
  // Domain 1: Data Engineering (6 questions)
  {
    id: 1,
    domain: "Data Engineering",
    question: "A machine learning team has several large CSV datasets in Amazon S3. Models built with the Amazon SageMaker Linear Learner algorithm have historically taken hours to train on similar-sized datasets. The team needs to accelerate the training process. What should a machine learning specialist do to address this concern?",
    options: [
      { id: "A", text: "Use Amazon SageMaker Pipe mode" },
      { id: "B", text: "Use Amazon Machine Learning to train the models" },
      { id: "C", text: "Use Amazon Kinesis to stream the data to Amazon SageMaker" },
      { id: "D", text: "Use AWS Glue to transform the CSV dataset to the JSON format" }
    ],
    correct: "A",
    explanations: {
      A: "CORRECT: Pipe mode streams data directly from S3 to the training algorithm without downloading to local storage first. This significantly reduces startup time and allows training to begin immediately while data continues streaming, which is ideal for large datasets.",
      B: "INCORRECT: Amazon Machine Learning is a legacy service that has been deprecated. Amazon SageMaker is the recommended service for ML workloads.",
      C: "INCORRECT: Amazon Kinesis is designed for real-time streaming data ingestion, not for training ML models on existing datasets stored in S3.",
      D: "INCORRECT: Converting CSV to JSON would not improve training speed. In fact, JSON is typically less efficient than CSV for tabular data due to additional formatting overhead."
    }
  },
  {
    id: 2,
    domain: "Data Engineering",
    question: "A company needs to ingest streaming data from IoT sensors, transform it, and store it for batch ML model training. The data arrives continuously at variable rates. Which architecture best meets these requirements?",
    options: [
      { id: "A", text: "Amazon API Gateway → AWS Lambda → Amazon S3" },
      { id: "B", text: "Amazon Kinesis Data Streams → Amazon Kinesis Data Firehose → Amazon S3" },
      { id: "C", text: "Amazon SQS → AWS Lambda → Amazon RDS" },
      { id: "D", text: "AWS Direct Connect → Amazon EC2 → Amazon EBS" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: API Gateway with Lambda has concurrency limits and is better suited for request-response patterns rather than continuous high-volume streaming data.",
      B: "CORRECT: Kinesis Data Streams handles variable-rate streaming ingestion with automatic scaling. Kinesis Data Firehose provides near-real-time delivery to S3 with optional transformation, and S3 is ideal for storing training data for batch ML workloads.",
      C: "INCORRECT: SQS is a message queue, not optimized for streaming analytics. RDS is a relational database not ideal for storing large volumes of ML training data.",
      D: "INCORRECT: Direct Connect is for dedicated network connections to AWS, not data ingestion. EC2 with EBS doesn't provide the managed streaming capabilities needed."
    }
  },
  {
    id: 3,
    domain: "Data Engineering",
    question: "A data engineer needs to create a data pipeline that extracts data from multiple sources, transforms it for ML feature engineering, and loads it into a data lake. The pipeline must be serverless and support both batch and micro-batch processing. Which service should be used?",
    options: [
      { id: "A", text: "Amazon EMR with Apache Spark" },
      { id: "B", text: "AWS Glue with PySpark" },
      { id: "C", text: "Amazon Redshift with stored procedures" },
      { id: "D", text: "Amazon Athena with CTAS queries" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: EMR is not serverless—it requires cluster management and provisioning. While it supports Spark, it doesn't meet the serverless requirement.",
      B: "CORRECT: AWS Glue is a fully serverless ETL service that supports PySpark for transformations, can handle both batch and streaming/micro-batch workloads, integrates with the Glue Data Catalog, and is ideal for ML feature engineering pipelines.",
      C: "INCORRECT: Redshift is a data warehouse, not an ETL tool. Stored procedures are for SQL operations within the warehouse, not for building data pipelines.",
      D: "INCORRECT: Athena is a query service for analyzing data in S3, not an ETL pipeline tool. CTAS can create tables but lacks the transformation capabilities needed for ML feature engineering."
    }
  },
  {
    id: 4,
    domain: "Data Engineering",
    question: "A company stores training data in Amazon S3 with sensitive PII that must be masked before ML training. The data is in Parquet format and updated daily. Which approach provides automated, scalable PII detection and masking?",
    options: [
      { id: "A", text: "Use Amazon Macie to detect PII and AWS Lambda to mask it" },
      { id: "B", text: "Use AWS Glue with the Sensitive Data Detection transform" },
      { id: "C", text: "Use Amazon Comprehend to detect PII and Amazon EMR to mask it" },
      { id: "D", text: "Use AWS Config rules to detect PII and SNS to alert for manual masking" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Macie detects sensitive data but is primarily for security monitoring and alerting, not for inline transformation of data. Combining with Lambda would require custom code and doesn't scale well for large Parquet files.",
      B: "CORRECT: AWS Glue's Sensitive Data Detection transform can automatically detect and mask PII within ETL jobs. It's serverless, scales automatically, works natively with Parquet, and can be scheduled to run daily as part of the data pipeline.",
      C: "INCORRECT: Comprehend can detect PII in text but requires additional orchestration. EMR is not serverless and adds operational overhead for this use case.",
      D: "INCORRECT: AWS Config is for resource compliance, not data content inspection. Manual masking doesn't meet the automated, scalable requirement."
    }
  },
  {
    id: 5,
    domain: "Data Engineering",
    question: "A data scientist needs to query and join multiple large datasets stored in Amazon S3 for exploratory analysis before feature engineering. The datasets are in CSV and Parquet formats. Which approach is most cost-effective for ad-hoc analysis?",
    options: [
      { id: "A", text: "Load all data into Amazon Redshift and run SQL queries" },
      { id: "B", text: "Use Amazon Athena with AWS Glue Data Catalog" },
      { id: "C", text: "Create an Amazon EMR cluster with Presto" },
      { id: "D", text: "Copy data to Amazon RDS and use standard SQL" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Redshift requires provisioning a cluster and loading data, which adds cost and time. Not ideal for ad-hoc exploratory analysis where you pay for running clusters.",
      B: "CORRECT: Athena is serverless and charges only per query based on data scanned. The Glue Data Catalog provides schema management. Athena natively supports both CSV and Parquet, making it ideal for cost-effective ad-hoc analysis.",
      C: "INCORRECT: EMR requires cluster management and you pay for running instances regardless of query activity. Not cost-effective for sporadic ad-hoc analysis.",
      D: "INCORRECT: RDS is not designed for analytical workloads on large datasets. Loading data into RDS would be slow and the storage costs would be higher than S3."
    }
  },
  {
    id: 6,
    domain: "Data Engineering",
    question: "A company wants to create a feature store for ML models. Features must be available for both real-time inference (low latency) and batch training. Which AWS service is purpose-built for this requirement?",
    options: [
      { id: "A", text: "Amazon DynamoDB with DynamoDB Streams" },
      { id: "B", text: "Amazon ElastiCache with Amazon S3" },
      { id: "C", text: "Amazon SageMaker Feature Store" },
      { id: "D", text: "Amazon Redshift with Redshift Spectrum" }
    ],
    correct: "C",
    explanations: {
      A: "INCORRECT: While DynamoDB provides low-latency access, building a feature store with DynamoDB requires significant custom development for feature versioning, lineage tracking, and batch access patterns.",
      B: "INCORRECT: This combination could work but requires custom implementation and doesn't provide feature store capabilities like versioning, metadata management, or integration with ML workflows.",
      C: "CORRECT: SageMaker Feature Store is purpose-built for ML feature management. It provides both an online store (low-latency real-time access) and an offline store (S3-based for batch training), with automatic synchronization, feature versioning, and native SageMaker integration.",
      D: "INCORRECT: Redshift is a data warehouse optimized for analytics, not for low-latency real-time feature serving required during inference."
    }
  },
  // Domain 2: Exploratory Data Analysis (7 questions)
  {
    id: 7,
    domain: "Exploratory Data Analysis",
    question: "A data scientist is analyzing a dataset and discovers that the target variable for a binary classification problem has 95% negative cases and 5% positive cases. Which technique is LEAST likely to help address this class imbalance?",
    options: [
      { id: "A", text: "SMOTE (Synthetic Minority Over-sampling Technique)" },
      { id: "B", text: "Adjusting class weights in the loss function" },
      { id: "C", text: "Undersampling the majority class" },
      { id: "D", text: "Increasing the learning rate" }
    ],
    correct: "D",
    explanations: {
      A: "INCORRECT (this DOES help): SMOTE generates synthetic examples of the minority class by interpolating between existing samples, helping balance the training data.",
      B: "INCORRECT (this DOES help): Adjusting class weights penalizes misclassification of the minority class more heavily, effectively making the model pay more attention to rare cases.",
      C: "INCORRECT (this DOES help): Undersampling reduces the majority class to balance with the minority class, though it may lose some information.",
      D: "CORRECT: Increasing the learning rate affects how quickly the model updates weights during training but does nothing to address class imbalance. It may actually make training unstable without addressing the underlying distribution issue."
    }
  },
  {
    id: 8,
    domain: "Exploratory Data Analysis",
    question: "A machine learning engineer is preparing features for a model and notices that one numerical feature has a highly right-skewed distribution with some extreme outliers. Which transformation would be most appropriate to normalize this distribution?",
    options: [
      { id: "A", text: "Min-max scaling" },
      { id: "B", text: "Log transformation" },
      { id: "C", text: "Z-score standardization" },
      { id: "D", text: "One-hot encoding" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Min-max scaling compresses values to a range (e.g., 0-1) but preserves the original distribution shape. Outliers would still dominate and skewness remains.",
      B: "CORRECT: Log transformation compresses the range of large values while spreading out smaller values, which is specifically effective at reducing right-skewness and mitigating the impact of outliers.",
      C: "INCORRECT: Z-score standardization centers data around mean with unit variance but doesn't change the distribution shape. Skewness and outlier impact remain.",
      D: "INCORRECT: One-hot encoding is for categorical variables, not numerical features."
    }
  },
  {
    id: 9,
    domain: "Exploratory Data Analysis",
    question: "A data scientist is building a term frequency-inverse document frequency (TF-IDF) matrix from a text corpus. The corpus contains 10,000 documents, and a specific term appears in 100 documents. What does the IDF component measure for this term?",
    options: [
      { id: "A", text: "How frequently the term appears within a single document" },
      { id: "B", text: "How rare or common the term is across the entire corpus" },
      { id: "C", text: "The total count of the term across all documents" },
      { id: "D", text: "The semantic similarity between documents containing the term" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: This describes Term Frequency (TF), not Inverse Document Frequency (IDF).",
      B: "CORRECT: IDF measures how rare or common a term is across the corpus. It's calculated as log(total documents / documents containing term). Rare terms get higher IDF scores, common terms get lower scores. This helps weight distinctive terms higher than common words like 'the' or 'is'.",
      C: "INCORRECT: Total count across documents is not what IDF measures. IDF cares about how many documents contain the term, not total occurrences.",
      D: "INCORRECT: TF-IDF doesn't measure semantic similarity. It's a statistical measure of term importance."
    }
  },
  {
    id: 10,
    domain: "Exploratory Data Analysis",
    question: "A dataset has 50 features for predicting customer churn. A data scientist wants to reduce dimensionality while preserving the maximum variance in the data. The scientist should NOT be concerned about feature interpretability. Which technique is most appropriate?",
    options: [
      { id: "A", text: "Recursive Feature Elimination (RFE)" },
      { id: "B", text: "Principal Component Analysis (PCA)" },
      { id: "C", text: "L1 regularization (Lasso)" },
      { id: "D", text: "Variance threshold feature selection" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: RFE is a feature selection method that keeps original features, which maintains interpretability. It doesn't maximize variance preservation as its primary objective.",
      B: "CORRECT: PCA creates new features (principal components) that are linear combinations of original features, specifically designed to maximize variance captured. Since interpretability is not a concern, PCA's transformed features are ideal.",
      C: "INCORRECT: L1 regularization performs feature selection by zeroing out coefficients, keeping original interpretable features. It doesn't transform features or explicitly maximize variance.",
      D: "INCORRECT: Variance threshold removes low-variance features but doesn't transform or combine features to maximize overall variance preservation."
    }
  },
  {
    id: 11,
    domain: "Exploratory Data Analysis",
    question: "A machine learning engineer notices that a numerical feature has 15% missing values. The data is missing completely at random (MCAR). The feature has a normal distribution with no outliers. Which imputation method is most appropriate?",
    options: [
      { id: "A", text: "Delete all rows with missing values" },
      { id: "B", text: "Impute with the mean of non-missing values" },
      { id: "C", text: "Impute with the mode of non-missing values" },
      { id: "D", text: "Impute with zero" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Deleting 15% of data loses significant information and reduces training set size substantially, especially when the data is MCAR and imputation is viable.",
      B: "CORRECT: For normally distributed data that is MCAR with no outliers, mean imputation is appropriate. It preserves the central tendency of the distribution without introducing bias.",
      C: "INCORRECT: Mode is appropriate for categorical variables, not continuous numerical features with a normal distribution.",
      D: "INCORRECT: Imputing with zero would shift the distribution and introduce bias, especially if zero is not a meaningful value for the feature."
    }
  },
  {
    id: 12,
    domain: "Exploratory Data Analysis",
    question: "A data scientist is analyzing feature correlations and finds two features with a Pearson correlation coefficient of 0.95. The scientist is building a linear regression model. What is the primary concern with including both features?",
    options: [
      { id: "A", text: "The model will underfit the training data" },
      { id: "B", text: "Multicollinearity will make coefficient estimates unstable" },
      { id: "C", text: "The model will be unable to converge" },
      { id: "D", text: "The training time will increase exponentially" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Multicollinearity doesn't cause underfitting. The model can still fit the data well; the issue is with interpreting individual feature contributions.",
      B: "CORRECT: High correlation (0.95) between features causes multicollinearity in linear regression. This makes coefficient estimates unstable and highly sensitive to small changes in data, inflates standard errors, and makes it difficult to determine individual feature importance.",
      C: "INCORRECT: Linear regression will still converge with correlated features (unless perfectly correlated at 1.0). Convergence is not the primary concern.",
      D: "INCORRECT: Two correlated features don't significantly impact training time. The computational cost remains linear with number of features."
    }
  },
  {
    id: 13,
    domain: "Exploratory Data Analysis",
    question: "A dataset contains a categorical feature 'Country' with 150 unique values. The feature is important for prediction but encoding all categories would create too many dimensions. Which encoding strategy best balances dimensionality and information preservation?",
    options: [
      { id: "A", text: "One-hot encoding all 150 categories" },
      { id: "B", text: "Label encoding with integer values 1-150" },
      { id: "C", text: "Target encoding (mean encoding) based on the target variable" },
      { id: "D", text: "Binary encoding with binary representations" }
    ],
    correct: "C",
    explanations: {
      A: "INCORRECT: One-hot encoding would create 150 sparse features, significantly increasing dimensionality and potentially causing issues with model training and overfitting.",
      B: "INCORRECT: Label encoding implies ordinal relationships between countries (e.g., Country 100 > Country 50), which is meaningless for nominal categorical data and can mislead the model.",
      C: "CORRECT: Target encoding replaces each category with the mean of the target variable for that category, reducing 150 categories to a single informative feature. It preserves predictive information while dramatically reducing dimensionality. (Note: requires careful handling of overfitting through techniques like smoothing or cross-validation.)",
      D: "INCORRECT: Binary encoding would still require log2(150) ≈ 8 features and doesn't preserve the relationship between categories and the target variable as effectively as target encoding."
    }
  },
  // Domain 3: Modeling (11 questions)
  {
    id: 14,
    domain: "Modeling",
    question: "A company wants to deploy a fraud detection model. Fraudulent transactions are rare (0.1% of data), and the business cost of missing a fraud is 100x higher than a false positive. Which evaluation metric should be prioritized?",
    options: [
      { id: "A", text: "Accuracy" },
      { id: "B", text: "Precision" },
      { id: "C", text: "Recall" },
      { id: "D", text: "F1 Score" }
    ],
    correct: "C",
    explanations: {
      A: "INCORRECT: With 99.9% non-fraud, a model predicting all transactions as legitimate would have 99.9% accuracy while catching zero frauds. Accuracy is misleading for imbalanced datasets.",
      B: "INCORRECT: Precision measures what percentage of flagged frauds are actual frauds. Optimizing for precision would reduce false positives but potentially miss more real frauds.",
      C: "CORRECT: Recall measures what percentage of actual frauds are caught. Given that missing a fraud costs 100x more than a false positive, maximizing recall (catching as many frauds as possible) aligns with the business objective.",
      D: "INCORRECT: F1 Score balances precision and recall equally, but the business clearly values recall more (100x cost difference), so F1 doesn't properly weight the business priorities."
    }
  },
  {
    id: 15,
    domain: "Modeling",
    question: "A data scientist is training a deep neural network and observes that training loss continues to decrease while validation loss starts increasing after epoch 10. What is occurring and what should be done?",
    options: [
      { id: "A", text: "Underfitting; increase model complexity" },
      { id: "B", text: "Overfitting; implement early stopping at epoch 10" },
      { id: "C", text: "Vanishing gradients; use ReLU activation" },
      { id: "D", text: "Learning rate is too low; increase learning rate" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Underfitting would show high training loss that doesn't decrease sufficiently. Here, training loss is decreasing, indicating the model is learning.",
      B: "CORRECT: Decreasing training loss with increasing validation loss is the classic sign of overfitting—the model is memorizing training data rather than learning generalizable patterns. Early stopping at epoch 10 (where validation loss was minimum) prevents further overfitting.",
      C: "INCORRECT: Vanishing gradients would cause training loss to plateau or decrease very slowly, not continue decreasing while validation loss increases.",
      D: "INCORRECT: The learning rate appears adequate since training loss is decreasing. This pattern is about generalization, not learning rate issues."
    }
  },
  {
    id: 16,
    domain: "Modeling",
    question: "A machine learning team needs to build a model to predict the exact number of products a customer will purchase next month (0, 1, 2, 3, etc.). Which type of model is most appropriate?",
    options: [
      { id: "A", text: "Binary classification model" },
      { id: "B", text: "Multi-class classification model" },
      { id: "C", text: "Linear regression model" },
      { id: "D", text: "Poisson regression model" }
    ],
    correct: "D",
    explanations: {
      A: "INCORRECT: Binary classification only predicts two outcomes (e.g., will/won't purchase), not the count of purchases.",
      B: "INCORRECT: While multi-class could predict discrete values, it treats each count as an independent category without recognizing that 3 > 2 > 1. It also struggles with unseen count values.",
      C: "INCORRECT: Linear regression can predict any real number, including negatives and fractions, which don't make sense for count data.",
      D: "CORRECT: Poisson regression is specifically designed for count data. It models non-negative integers, accounts for the discrete nature of counts, and handles the typically right-skewed distribution of count data appropriately."
    }
  },
  {
    id: 17,
    domain: "Modeling",
    question: "An ML engineer is using Amazon SageMaker's built-in XGBoost algorithm. The model is overfitting. Which hyperparameter adjustment would MOST likely reduce overfitting?",
    options: [
      { id: "A", text: "Increase max_depth from 6 to 12" },
      { id: "B", text: "Increase num_round from 100 to 500" },
      { id: "C", text: "Decrease min_child_weight from 1 to 0.5" },
      { id: "D", text: "Decrease subsample from 1.0 to 0.8" }
    ],
    correct: "D",
    explanations: {
      A: "INCORRECT: Increasing max_depth allows deeper trees that can capture more complex patterns, which increases overfitting risk.",
      B: "INCORRECT: Increasing num_round (number of boosting rounds) adds more trees, which can increase overfitting by continuing to fit noise in the training data.",
      C: "INCORRECT: Decreasing min_child_weight allows splits with fewer samples, creating more specific rules that increase overfitting.",
      D: "CORRECT: Decreasing subsample from 1.0 to 0.8 means each tree trains on a random 80% subset of data, introducing regularization through randomness and reducing overfitting. This is a standard technique to improve generalization."
    }
  },
  {
    id: 18,
    domain: "Modeling",
    question: "A company wants to build a recommendation system for an e-commerce platform. They have historical user-item interaction data (purchases, views, ratings) but no content metadata about products. Which approach is most suitable?",
    options: [
      { id: "A", text: "Content-based filtering" },
      { id: "B", text: "Collaborative filtering" },
      { id: "C", text: "Knowledge-based recommendations" },
      { id: "D", text: "Rule-based recommendations" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Content-based filtering requires product metadata (descriptions, categories, attributes) to find similar items. The scenario explicitly states no content metadata is available.",
      B: "CORRECT: Collaborative filtering uses user-item interaction patterns to make recommendations, finding similar users or items based purely on behavior data. It works with exactly what's available: purchases, views, and ratings.",
      C: "INCORRECT: Knowledge-based systems require explicit product knowledge and rules about product features and user requirements, which aren't available here.",
      D: "INCORRECT: Rule-based recommendations require manually defined rules and don't leverage the interaction data effectively at scale."
    }
  },
  {
    id: 19,
    domain: "Modeling",
    question: "A data scientist is using Amazon SageMaker's built-in algorithm to detect anomalies in streaming IoT sensor data. Which algorithm is specifically designed for anomaly detection?",
    options: [
      { id: "A", text: "XGBoost" },
      { id: "B", text: "Random Cut Forest (RCF)" },
      { id: "C", text: "Linear Learner" },
      { id: "D", text: "K-Means" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: XGBoost is a supervised learning algorithm for classification and regression, not specifically designed for unsupervised anomaly detection.",
      B: "CORRECT: Random Cut Forest is SageMaker's built-in unsupervised algorithm specifically designed for anomaly detection. It works well with streaming data and assigns anomaly scores to data points based on how much they change the forest structure.",
      C: "INCORRECT: Linear Learner is for supervised classification and regression problems, not unsupervised anomaly detection.",
      D: "INCORRECT: K-Means is for clustering, grouping similar data points together. While anomalies might fall outside clusters, it's not specifically designed or optimized for anomaly detection like RCF."
    }
  },
  {
    id: 20,
    domain: "Modeling",
    question: "A machine learning engineer is training a neural network for image classification. The training process is slow and unstable with loss values oscillating. Which technique would MOST likely help stabilize and speed up training?",
    options: [
      { id: "A", text: "Remove all hidden layers" },
      { id: "B", text: "Use batch normalization" },
      { id: "C", text: "Increase the learning rate by 10x" },
      { id: "D", text: "Use a smaller batch size of 1" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Removing hidden layers would cripple the model's ability to learn complex patterns needed for image classification.",
      B: "CORRECT: Batch normalization normalizes layer inputs, reducing internal covariate shift. This stabilizes training, allows higher learning rates, reduces sensitivity to initialization, and often speeds up convergence significantly.",
      C: "INCORRECT: If training is already oscillating, increasing the learning rate would likely make it worse or cause divergence.",
      D: "INCORRECT: Batch size of 1 (stochastic gradient descent) typically increases noise in gradient estimates, making training less stable, not more stable."
    }
  },
  {
    id: 21,
    domain: "Modeling",
    question: "A data scientist needs to perform hyperparameter tuning for a SageMaker training job. The hyperparameter search space is large with many interdependent parameters. Which SageMaker hyperparameter tuning strategy would be most efficient?",
    options: [
      { id: "A", text: "Grid search" },
      { id: "B", text: "Random search" },
      { id: "C", text: "Bayesian optimization" },
      { id: "D", text: "Manual tuning" }
    ],
    correct: "C",
    explanations: {
      A: "INCORRECT: Grid search exhaustively tries all combinations, which is computationally prohibitive with large search spaces and many parameters.",
      B: "INCORRECT: Random search is more efficient than grid search but doesn't learn from previous trials. It treats each trial independently.",
      C: "CORRECT: Bayesian optimization builds a probabilistic model of the objective function, learning from previous evaluations to intelligently select the next hyperparameters to try. This is especially effective for expensive evaluations and can find good solutions with fewer trials.",
      D: "INCORRECT: Manual tuning is time-consuming, doesn't scale, and relies on human intuition rather than systematic optimization."
    }
  },
  {
    id: 22,
    domain: "Modeling",
    question: "A company is building a natural language model to classify customer support tickets into 50 different categories. They have limited labeled training data. Which approach would be most effective?",
    options: [
      { id: "A", text: "Train a custom transformer model from scratch" },
      { id: "B", text: "Use transfer learning with a pre-trained language model like BERT" },
      { id: "C", text: "Use a simple bag-of-words model with logistic regression" },
      { id: "D", text: "Train a recurrent neural network from scratch" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Training transformers from scratch requires massive amounts of data and compute. With limited labeled data, this would severely underperform.",
      B: "CORRECT: Transfer learning with pre-trained models like BERT leverages knowledge learned from large text corpora. Fine-tuning on limited domain-specific data is highly effective because the model already understands language patterns.",
      C: "INCORRECT: Bag-of-words loses word order and semantic meaning. With 50 categories and limited data, this simple approach would likely struggle with nuanced classification.",
      D: "INCORRECT: Training RNNs from scratch also requires substantial data. Without pre-training, the model must learn language representations and task-specific patterns simultaneously."
    }
  },
  {
    id: 23,
    domain: "Modeling",
    question: "An ML engineer is building an object detection model and needs to evaluate its performance. The model outputs bounding boxes around detected objects. Which metric is most appropriate for evaluating this model?",
    options: [
      { id: "A", text: "Accuracy" },
      { id: "B", text: "Mean Average Precision (mAP)" },
      { id: "C", text: "Root Mean Square Error (RMSE)" },
      { id: "D", text: "Area Under the ROC Curve (AUC-ROC)" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: Accuracy doesn't account for localization quality (how well bounding boxes match ground truth) or handle varying numbers of detections per image.",
      B: "CORRECT: mAP is the standard metric for object detection. It considers both classification accuracy AND localization quality (via Intersection over Union thresholds), averaged across all classes. It handles multiple detections and varying confidence thresholds.",
      C: "INCORRECT: RMSE is for regression tasks measuring continuous value prediction error, not appropriate for object detection evaluation.",
      D: "INCORRECT: AUC-ROC is for binary classification probability ranking, not for evaluating spatial localization in object detection."
    }
  },
  {
    id: 24,
    domain: "Modeling",
    question: "A data scientist is building a time series forecasting model to predict daily sales 30 days into the future. The data shows strong weekly seasonality and an upward trend. Which model type is most appropriate?",
    options: [
      { id: "A", text: "Simple exponential smoothing" },
      { id: "B", text: "ARIMA(1,1,1)" },
      { id: "C", text: "Prophet or SARIMA with weekly seasonality" },
      { id: "D", text: "Simple moving average" }
    ],
    correct: "C",
    explanations: {
      A: "INCORRECT: Simple exponential smoothing doesn't handle seasonality. It only captures level changes, missing the weekly patterns entirely.",
      B: "INCORRECT: Basic ARIMA doesn't explicitly model seasonality. While it handles trends (the 'd' differencing), it would miss the weekly patterns without the seasonal component.",
      C: "CORRECT: Both Prophet and SARIMA (Seasonal ARIMA) explicitly model seasonality and trends. Prophet is particularly good at multiple seasonalities and is robust to missing data. SARIMA adds seasonal terms to capture weekly patterns.",
      D: "INCORRECT: Simple moving average is backward-looking and doesn't forecast well, especially for 30 days ahead. It doesn't model trend or seasonality."
    }
  },
  // Domain 4: ML Implementation and Operations (6 questions)
  {
    id: 25,
    domain: "ML Implementation and Operations",
    question: "A company deployed an ML model to production 6 months ago. Model performance has degraded significantly. Investigation shows the input data distribution has shifted from training data. What is this phenomenon called, and what is the recommended solution?",
    options: [
      { id: "A", text: "Concept drift; retrain on new labeled data" },
      { id: "B", text: "Overfitting; reduce model complexity" },
      { id: "C", text: "Underfitting; increase model complexity" },
      { id: "D", text: "Data leakage; fix the training pipeline" }
    ],
    correct: "A",
    explanations: {
      A: "CORRECT: When input data distribution shifts from what the model was trained on, this is called concept drift (or data drift). The model's learned patterns no longer match reality. Retraining on recent labeled data that reflects current distributions is the standard solution.",
      B: "INCORRECT: Overfitting would have been apparent during initial validation, not emerge 6 months later. Overfitting is about train/validation gap, not production degradation over time.",
      C: "INCORRECT: Underfitting also would be apparent during training, not emerge later. The issue described is about data distribution change, not model capacity.",
      D: "INCORRECT: Data leakage causes artificially good training metrics that don't hold in production, but performance degradation over time with distribution shift is concept drift."
    }
  },
  {
    id: 26,
    domain: "ML Implementation and Operations",
    question: "A company needs to deploy a real-time ML inference endpoint that can handle variable traffic with occasional spikes. Cost optimization is important during low-traffic periods. Which SageMaker deployment option is most appropriate?",
    options: [
      { id: "A", text: "SageMaker real-time inference with a single ml.m5.xlarge instance" },
      { id: "B", text: "SageMaker real-time inference with auto-scaling configured" },
      { id: "C", text: "SageMaker batch transform" },
      { id: "D", text: "SageMaker Serverless Inference" }
    ],
    correct: "D",
    explanations: {
      A: "INCORRECT: A single fixed instance doesn't handle traffic spikes and still incurs costs during low-traffic periods.",
      B: "INCORRECT: Auto-scaling helps with spikes but has minimum instance requirements (at least 1 running), so you still pay during idle periods.",
      C: "INCORRECT: Batch transform is for processing large datasets offline, not real-time inference with immediate responses.",
      D: "CORRECT: SageMaker Serverless Inference automatically scales to zero during idle periods (pay nothing when no requests) and scales up instantly for traffic spikes. It's ideal for variable, unpredictable traffic patterns where cost optimization matters."
    }
  },
  {
    id: 27,
    domain: "ML Implementation and Operations",
    question: "An ML team wants to implement CI/CD for their machine learning models. They need to automatically test models, check for performance regression, and deploy approved models. Which combination of AWS services best supports this MLOps workflow?",
    options: [
      { id: "A", text: "Amazon S3, AWS Lambda, Amazon CloudWatch" },
      { id: "B", text: "SageMaker Pipelines, SageMaker Model Registry, CodePipeline" },
      { id: "C", text: "Amazon EC2, AWS Batch, Amazon SNS" },
      { id: "D", text: "AWS Glue, Amazon Athena, Amazon QuickSight" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: While these are useful services, they don't provide ML-specific CI/CD capabilities like model versioning, approval workflows, or ML pipeline orchestration.",
      B: "CORRECT: SageMaker Pipelines orchestrates ML workflows (training, evaluation). Model Registry provides versioning, metadata tracking, and approval status. CodePipeline integrates with these for automated deployment when models are approved—a complete MLOps CI/CD solution.",
      C: "INCORRECT: EC2 and Batch are compute services without ML-specific CI/CD features. SNS is for notifications, not pipeline orchestration.",
      D: "INCORRECT: Glue, Athena, and QuickSight are for data processing and visualization, not ML model deployment and CI/CD."
    }
  },
  {
    id: 28,
    domain: "ML Implementation and Operations",
    question: "A machine learning model in production needs to serve inference requests with strict latency requirements (< 50ms p99). The model is currently hosted on SageMaker and latency is around 200ms. Which approach would MOST effectively reduce inference latency?",
    options: [
      { id: "A", text: "Increase the instance count for the endpoint" },
      { id: "B", text: "Use SageMaker Neo to compile and optimize the model" },
      { id: "C", text: "Switch from real-time inference to batch transform" },
      { id: "D", text: "Add more training data and retrain the model" }
    ],
    correct: "B",
    explanations: {
      A: "INCORRECT: More instances help with throughput and handling more concurrent requests, but don't reduce the latency of individual inference calls.",
      B: "CORRECT: SageMaker Neo compiles models to optimized machine code for the target hardware, reducing inference time. It can achieve 2-10x performance improvement through graph optimizations, operator fusion, and hardware-specific optimizations.",
      C: "INCORRECT: Batch transform is for offline processing, not real-time low-latency inference. It would increase latency, not reduce it.",
      D: "INCORRECT: More training data might improve model accuracy but doesn't reduce inference latency. The model architecture and optimization determine latency."
    }
  },
  {
    id: 29,
    domain: "ML Implementation and Operations",
    question: "A company needs to ensure their deployed ML model doesn't discriminate based on protected attributes (gender, race, age). Which AWS service provides bias detection capabilities for ML models?",
    options: [
      { id: "A", text: "Amazon Macie" },
      { id: "B", text: "AWS Config" },
      { id: "C", text: "Amazon SageMaker Clarify" },
      { id: "D", text: "Amazon Detective" }
    ],
    correct: "C",
    explanations: {
      A: "INCORRECT: Amazon Macie is for discovering and protecting sensitive data (like PII) in S3, not for ML model bias detection.",
      B: "INCORRECT: AWS Config monitors resource configurations for compliance, not ML model fairness or bias.",
      C: "CORRECT: SageMaker Clarify provides bias detection during training and inference. It measures various fairness metrics (demographic parity, equalized odds, etc.) and generates reports showing potential bias across protected attributes.",
      D: "INCORRECT: Amazon Detective is for security investigation and analyzing security findings, not ML fairness or bias detection."
    }
  },
  {
    id: 30,
    domain: "ML Implementation and Operations",
    question: "A data scientist wants to understand why a deployed SageMaker model made a specific prediction for a customer loan application. The model uses a complex gradient boosting algorithm. Which SageMaker capability should be used?",
    options: [
      { id: "A", text: "SageMaker Debugger" },
      { id: "B", text: "SageMaker Model Monitor" },
      { id: "C", text: "SageMaker Clarify feature attributions (SHAP)" },
      { id: "D", text: "SageMaker Experiments" }
    ],
    correct: "C",
    explanations: {
      A: "INCORRECT: SageMaker Debugger is for debugging training jobs—analyzing tensors, detecting training issues like vanishing gradients. It doesn't explain individual predictions.",
      B: "INCORRECT: Model Monitor tracks data quality, bias drift, and model quality over time. It monitors overall model behavior, not individual prediction explanations.",
      C: "CORRECT: SageMaker Clarify uses SHAP (SHapley Additive exPlanations) to provide feature attributions for individual predictions. It shows which features contributed most to a specific prediction and in what direction—essential for explainable AI in regulated domains like lending.",
      D: "INCORRECT: SageMaker Experiments tracks and compares training runs and hyperparameters. It's for experiment management, not prediction explainability."
    }
  }
];

const formatTime = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

export default function AWSMLQuiz() {
  const [screen, setScreen] = useState('start');
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  const [elapsedTime, setElapsedTime] = useState(0);
  const [finalTime, setFinalTime] = useState(0);
  const timerRef = useRef(null);

  useEffect(() => {
    if (screen === 'test') {
      timerRef.current = setInterval(() => {
        setElapsedTime(prev => prev + 1);
      }, 1000);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [screen]);

  const startTest = () => {
    setScreen('test');
    setCurrentQuestion(0);
    setAnswers({});
    setElapsedTime(0);
  };

  const handleAnswer = (questionId, answerId) => {
    setAnswers(prev => ({ ...prev, [questionId]: answerId }));
  };

  const finishTest = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setFinalTime(elapsedTime);
    setScreen('results');
  };

  const calculateScore = () => {
    let correct = 0;
    questions.forEach(q => {
      if (answers[q.id] === q.correct) correct++;
    });
    return correct;
  };

  const startReview = () => {
    setCurrentQuestion(0);
    setScreen('review');
  };

  const restartTest = () => {
    setScreen('start');
    setAnswers({});
    setElapsedTime(0);
    setFinalTime(0);
    setCurrentQuestion(0);
  };

  const q = questions[currentQuestion];
  const score = calculateScore();
  const percentage = Math.round((score / questions.length) * 100);
  const passed = percentage >= 75;

  // Domain breakdown for results
  const domainScores = {};
  questions.forEach(question => {
    if (!domainScores[question.domain]) {
      domainScores[question.domain] = { correct: 0, total: 0 };
    }
    domainScores[question.domain].total++;
    if (answers[question.id] === question.correct) {
      domainScores[question.domain].correct++;
    }
  });

  if (screen === 'start') {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-6">
        <div className="max-w-2xl w-full bg-slate-800 rounded-2xl shadow-2xl p-8 border border-slate-700">
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-20 h-20 bg-gradient-to-br from-orange-500 to-amber-600 rounded-2xl mb-6 shadow-lg">
              <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h1 className="text-3xl font-bold text-white mb-2">AWS Machine Learning Specialty</h1>
            <p className="text-slate-400 text-lg">Practice Exam (MLS-C01)</p>
          </div>
          
          <div className="bg-slate-700/50 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-semibold text-white mb-4">Exam Overview</h2>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div className="flex items-center text-slate-300">
                <svg className="w-5 h-5 mr-2 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                30 Questions
              </div>
              <div className="flex items-center text-slate-300">
                <svg className="w-5 h-5 mr-2 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                75% to Pass
              </div>
              <div className="flex items-center text-slate-300">
                <svg className="w-5 h-5 mr-2 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Timed (Count-up)
              </div>
              <div className="flex items-center text-slate-300">
                <svg className="w-5 h-5 mr-2 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                </svg>
                Navigate Freely
              </div>
            </div>
          </div>

          <div className="bg-slate-700/50 rounded-xl p-6 mb-8">
            <h2 className="text-lg font-semibold text-white mb-3">Domain Coverage</h2>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between text-slate-300">
                <span>Data Engineering</span>
                <span className="text-amber-500">6 questions (20%)</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Exploratory Data Analysis</span>
                <span className="text-amber-500">7 questions (24%)</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Modeling</span>
                <span className="text-amber-500">11 questions (36%)</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>ML Implementation & Operations</span>
                <span className="text-amber-500">6 questions (20%)</span>
              </div>
            </div>
          </div>

          <button
            onClick={startTest}
            className="w-full py-4 bg-gradient-to-r from-orange-500 to-amber-600 text-white font-semibold rounded-xl hover:from-orange-600 hover:to-amber-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5"
          >
            Start Exam
          </button>
        </div>
      </div>
    );
  }

  if (screen === 'test') {
    const answeredCount = Object.keys(answers).length;
    
    return (
      <div className="min-h-screen bg-slate-900 p-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="bg-slate-800 rounded-xl p-4 mb-6 flex items-center justify-between border border-slate-700">
            <div className="flex items-center space-x-6">
              <div className="text-slate-400 text-sm">
                Question <span className="text-white font-semibold">{currentQuestion + 1}</span> of {questions.length}
              </div>
              <div className="text-slate-400 text-sm">
                Answered: <span className="text-amber-500 font-semibold">{answeredCount}</span>/{questions.length}
              </div>
            </div>
            <div className="flex items-center space-x-2 bg-slate-700 px-4 py-2 rounded-lg">
              <svg className="w-5 h-5 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="text-white font-mono font-semibold">{formatTime(elapsedTime)}</span>
            </div>
          </div>

          {/* Progress bar */}
          <div className="h-2 bg-slate-700 rounded-full mb-6 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-orange-500 to-amber-500 transition-all duration-300"
              style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
            />
          </div>

          {/* Question Card */}
          <div className="bg-slate-800 rounded-2xl p-8 mb-6 border border-slate-700">
            <div className="inline-block px-3 py-1 bg-amber-500/20 text-amber-500 text-sm font-medium rounded-full mb-4">
              {q.domain}
            </div>
            <h2 className="text-xl text-white mb-8 leading-relaxed">{q.question}</h2>
            
            <div className="space-y-3">
              {q.options.map(option => (
                <label
                  key={option.id}
                  className={`flex items-start p-4 rounded-xl cursor-pointer transition-all duration-200 border-2 ${
                    answers[q.id] === option.id
                      ? 'border-amber-500 bg-amber-500/10'
                      : 'border-slate-600 hover:border-slate-500 hover:bg-slate-700/50'
                  }`}
                >
                  <input
                    type="radio"
                    name={`question-${q.id}`}
                    checked={answers[q.id] === option.id}
                    onChange={() => handleAnswer(q.id, option.id)}
                    className="sr-only"
                  />
                  <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center font-semibold mr-4 ${
                    answers[q.id] === option.id
                      ? 'bg-amber-500 text-white'
                      : 'bg-slate-600 text-slate-300'
                  }`}>
                    {option.id}
                  </div>
                  <span className="text-slate-200 pt-1">{option.text}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setCurrentQuestion(prev => Math.max(0, prev - 1))}
              disabled={currentQuestion === 0}
              className="flex items-center px-6 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            >
              <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Previous
            </button>

            <div className="flex items-center space-x-2">
              {questions.map((_, idx) => (
                <button
                  key={idx}
                  onClick={() => setCurrentQuestion(idx)}
                  className={`w-3 h-3 rounded-full transition-all ${
                    idx === currentQuestion
                      ? 'bg-amber-500 scale-125'
                      : answers[questions[idx].id]
                      ? 'bg-green-500'
                      : 'bg-slate-600 hover:bg-slate-500'
                  }`}
                />
              ))}
            </div>

            {currentQuestion === questions.length - 1 ? (
              <button
                onClick={finishTest}
                className="flex items-center px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-xl hover:from-green-600 hover:to-emerald-700 transition-all shadow-lg"
              >
                Finish Exam
                <svg className="w-5 h-5 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </button>
            ) : (
              <button
                onClick={() => setCurrentQuestion(prev => Math.min(questions.length - 1, prev + 1))}
                className="flex items-center px-6 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition-all"
              >
                Next
                <svg className="w-5 h-5 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (screen === 'results') {
    return (
      <div className="min-h-screen bg-slate-900 p-6">
        <div className="max-w-2xl mx-auto">
          <div className="bg-slate-800 rounded-2xl p-8 border border-slate-700 text-center mb-6">
            <div className={`inline-flex items-center justify-center w-24 h-24 rounded-full mb-6 ${
              passed ? 'bg-green-500/20' : 'bg-red-500/20'
            }`}>
              {passed ? (
                <svg className="w-12 h-12 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ) : (
                <svg className="w-12 h-12 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </div>
            
            <h1 className={`text-3xl font-bold mb-2 ${passed ? 'text-green-500' : 'text-red-500'}`}>
              {passed ? 'PASSED' : 'NOT PASSED'}
            </h1>
            <p className="text-slate-400 mb-8">Practice Exam Complete</p>

            <div className="grid grid-cols-3 gap-4 mb-8">
              <div className="bg-slate-700/50 rounded-xl p-4">
                <div className="text-3xl font-bold text-white">{score}/{questions.length}</div>
                <div className="text-slate-400 text-sm">Correct</div>
              </div>
              <div className="bg-slate-700/50 rounded-xl p-4">
                <div className={`text-3xl font-bold ${passed ? 'text-green-500' : 'text-red-500'}`}>{percentage}%</div>
                <div className="text-slate-400 text-sm">Score</div>
              </div>
              <div className="bg-slate-700/50 rounded-xl p-4">
                <div className="text-3xl font-bold text-amber-500">{formatTime(finalTime)}</div>
                <div className="text-slate-400 text-sm">Time</div>
              </div>
            </div>

            <div className="bg-slate-700/50 rounded-xl p-6 text-left">
              <h2 className="text-lg font-semibold text-white mb-4">Domain Breakdown</h2>
              <div className="space-y-3">
                {Object.entries(domainScores).map(([domain, scores]) => {
                  const domainPct = Math.round((scores.correct / scores.total) * 100);
                  return (
                    <div key={domain}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-slate-300">{domain}</span>
                        <span className={domainPct >= 75 ? 'text-green-500' : 'text-red-500'}>
                          {scores.correct}/{scores.total} ({domainPct}%)
                        </span>
                      </div>
                      <div className="h-2 bg-slate-600 rounded-full overflow-hidden">
                        <div 
                          className={`h-full transition-all ${domainPct >= 75 ? 'bg-green-500' : 'bg-red-500'}`}
                          style={{ width: `${domainPct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          <div className="flex space-x-4">
            <button
              onClick={startReview}
              className="flex-1 py-4 bg-gradient-to-r from-orange-500 to-amber-600 text-white font-semibold rounded-xl hover:from-orange-600 hover:to-amber-700 transition-all shadow-lg"
            >
              Review Answers
            </button>
            <button
              onClick={restartTest}
              className="flex-1 py-4 bg-slate-700 text-white font-semibold rounded-xl hover:bg-slate-600 transition-all"
            >
              Retake Exam
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (screen === 'review') {
    const userAnswer = answers[q.id];
    const isCorrect = userAnswer === q.correct;
    
    return (
      <div className="min-h-screen bg-slate-900 p-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="bg-slate-800 rounded-xl p-4 mb-6 flex items-center justify-between border border-slate-700">
            <div className="flex items-center space-x-4">
              <span className="text-slate-400 text-sm">
                Review Question <span className="text-white font-semibold">{currentQuestion + 1}</span> of {questions.length}
              </span>
              <span className={`px-3 py-1 rounded-full text-sm font-medium ${
                isCorrect ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'
              }`}>
                {isCorrect ? 'Correct' : 'Incorrect'}
              </span>
            </div>
            <button
              onClick={() => setScreen('results')}
              className="text-slate-400 hover:text-white transition-colors"
            >
              Back to Results
            </button>
          </div>

          {/* Question Card */}
          <div className="bg-slate-800 rounded-2xl p-8 mb-6 border border-slate-700">
            <div className="inline-block px-3 py-1 bg-amber-500/20 text-amber-500 text-sm font-medium rounded-full mb-4">
              {q.domain}
            </div>
            <h2 className="text-xl text-white mb-8 leading-relaxed">{q.question}</h2>
            
            <div className="space-y-4">
              {q.options.map(option => {
                const isThisCorrect = option.id === q.correct;
                const wasSelected = userAnswer === option.id;
                
                let borderColor = 'border-slate-600';
                let bgColor = '';
                
                if (isThisCorrect) {
                  borderColor = 'border-green-500';
                  bgColor = 'bg-green-500/10';
                } else if (wasSelected && !isThisCorrect) {
                  borderColor = 'border-red-500';
                  bgColor = 'bg-red-500/10';
                }
                
                return (
                  <div
                    key={option.id}
                    className={`p-4 rounded-xl border-2 ${borderColor} ${bgColor}`}
                  >
                    <div className="flex items-start mb-3">
                      <div className={`flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center font-semibold mr-4 ${
                        isThisCorrect
                          ? 'bg-green-500 text-white'
                          : wasSelected
                          ? 'bg-red-500 text-white'
                          : 'bg-slate-600 text-slate-300'
                      }`}>
                        {option.id}
                      </div>
                      <div className="flex-1">
                        <span className="text-slate-200">{option.text}</span>
                        {wasSelected && (
                          <span className="ml-2 text-sm text-slate-400">(Your answer)</span>
                        )}
                        {isThisCorrect && (
                          <span className="ml-2 text-sm text-green-500">(Correct answer)</span>
                        )}
                      </div>
                    </div>
                    <div className={`ml-12 text-sm p-3 rounded-lg ${
                      isThisCorrect ? 'bg-green-500/10 text-green-400' : 'bg-slate-700/50 text-slate-400'
                    }`}>
                      {q.explanations[option.id]}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between">
            <button
              onClick={() => setCurrentQuestion(prev => Math.max(0, prev - 1))}
              disabled={currentQuestion === 0}
              className="flex items-center px-6 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            >
              <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              Previous
            </button>

            <div className="flex items-center space-x-2">
              {questions.map((question, idx) => {
                const wasCorrect = answers[question.id] === question.correct;
                return (
                  <button
                    key={idx}
                    onClick={() => setCurrentQuestion(idx)}
                    className={`w-3 h-3 rounded-full transition-all ${
                      idx === currentQuestion
                        ? 'scale-125 ring-2 ring-white ring-offset-2 ring-offset-slate-900'
                        : ''
                    } ${wasCorrect ? 'bg-green-500' : 'bg-red-500'}`}
                  />
                );
              })}
            </div>

            <button
              onClick={() => setCurrentQuestion(prev => Math.min(questions.length - 1, prev + 1))}
              disabled={currentQuestion === questions.length - 1}
              className="flex items-center px-6 py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all"
            >
              Next
              <svg className="w-5 h-5 ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    );
  }

  return null;
}
