import React, { useState, useEffect, useRef, useCallback } from 'react';
import ProfileSelect from './components/ProfileSelect';
import TestDashboard from './components/TestDashboard';
import StudyGuide from './components/StudyGuide';
import {
  getActiveProfile,
  createAttempt,
  updateAttemptProgress,
  updateAttemptBookmarks,
  completeAttempt,
  getAttemptById,
  exportAsJSON,
  exportAsCSV,
  exportAsHTML
} from './storage';

const testBank = {
  test1: {
    name: "Practice Exam 1",
    description: "Core concepts across all domains",
    questions: [
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
    ]
  },
  test2: {
    name: "Practice Exam 2",
    description: "Advanced scenarios and edge cases",
    questions: [
      // Domain 1: Data Engineering (6 questions)
      {
        id: 1,
        domain: "Data Engineering",
        question: "A company has 500TB of historical log data in Amazon S3 that needs to be processed for ML training. The processing requires complex transformations using Python libraries not available in AWS Glue. Processing must complete within 4 hours. Which solution is most appropriate?",
        options: [
          { id: "A", text: "AWS Lambda with increased memory and timeout" },
          { id: "B", text: "Amazon EMR with custom bootstrap actions to install libraries" },
          { id: "C", text: "AWS Glue with custom Python wheel files" },
          { id: "D", text: "Amazon EC2 instances with cron jobs" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Lambda has a 15-minute timeout limit and 10GB memory limit. Processing 500TB within these constraints is not feasible.",
          B: "CORRECT: EMR can process massive datasets within time constraints using distributed Spark. Bootstrap actions allow installing any custom Python libraries. EMR auto-scales and provides the compute power needed for 500TB in 4 hours.",
          C: "INCORRECT: While Glue supports custom packages, it has limitations on package sizes and some native libraries. EMR provides more flexibility for complex custom dependencies.",
          D: "INCORRECT: EC2 instances would require manual orchestration, scaling, and wouldn't efficiently process 500TB in 4 hours without significant infrastructure management."
        }
      },
      {
        id: 2,
        domain: "Data Engineering",
        question: "A data pipeline processes clickstream data every hour, storing results in S3. Occasionally, late-arriving data needs to reprocess specific hours. The pipeline uses AWS Glue. How should the engineer handle idempotent reprocessing?",
        options: [
          { id: "A", text: "Delete all S3 data and reprocess from the beginning" },
          { id: "B", text: "Use Glue job bookmarks with time-based partitioning" },
          { id: "C", text: "Overwrite S3 partitions using dynamic partition overwrite mode" },
          { id: "D", text: "Append new data with a version column" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Reprocessing all data is inefficient and time-consuming. It doesn't provide targeted reprocessing capability.",
          B: "INCORRECT: Job bookmarks track what data has been processed to avoid reprocessing. They don't help when you WANT to reprocess specific time periods.",
          C: "CORRECT: Dynamic partition overwrite mode replaces only the specific partitions being written, leaving other partitions unchanged. This enables idempotent reprocessing—running the same hour multiple times produces the same result without duplicates.",
          D: "INCORRECT: Appending with versions creates duplicates and requires downstream logic to handle deduplication, adding complexity."
        }
      },
      {
        id: 3,
        domain: "Data Engineering",
        question: "A company needs to build a real-time ML inference system that enriches incoming events with features from a data lake before scoring. Features are updated daily, and inference latency must be under 100ms. Which architecture is most appropriate?",
        options: [
          { id: "A", text: "Query Amazon Athena for features during each inference request" },
          { id: "B", text: "Load daily features into Amazon ElastiCache and query during inference" },
          { id: "C", text: "Use SageMaker Feature Store with the online store" },
          { id: "D", text: "Store features in S3 and use S3 Select for fast retrieval" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Athena queries have cold start latency and per-query overhead. Sub-100ms latency is not reliably achievable for real-time inference.",
          B: "INCORRECT: While ElastiCache provides low latency, it requires custom implementation for feature management, versioning, and synchronization with the data lake.",
          C: "CORRECT: SageMaker Feature Store's online store is designed for low-latency feature retrieval (single-digit milliseconds). The offline store syncs with your data lake, and features are automatically promoted to the online store. It's purpose-built for this exact use case.",
          D: "INCORRECT: S3 Select still has latency overhead and isn't designed for real-time sub-100ms retrieval patterns typical in ML inference."
        }
      },
      {
        id: 4,
        domain: "Data Engineering",
        question: "A media company processes video files to extract frames for ML training. Videos are uploaded to S3 and must be processed within 5 minutes. Frame extraction requires FFmpeg and GPU acceleration. Which service should be used?",
        options: [
          { id: "A", text: "AWS Lambda with container images" },
          { id: "B", text: "AWS Batch with GPU instances" },
          { id: "C", text: "Amazon Elastic Transcoder" },
          { id: "D", text: "AWS Glue with Python shell jobs" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Lambda doesn't support GPU instances, and there are limitations on FFmpeg and video processing within Lambda's constraints.",
          B: "CORRECT: AWS Batch supports GPU instances (P3, G4), can run containerized FFmpeg with GPU acceleration, automatically scales based on queue depth, and can process videos quickly. It's ideal for compute-intensive media processing.",
          C: "INCORRECT: Elastic Transcoder is for video transcoding (format conversion), not frame extraction for ML training datasets.",
          D: "INCORRECT: Glue Python shell jobs don't support GPU instances and aren't designed for video processing workloads."
        }
      },
      {
        id: 5,
        domain: "Data Engineering",
        question: "A data lake contains 10 years of transaction data in Parquet format, partitioned by year/month/day. Most ML queries only access the last 30 days. How can query performance and cost be optimized?",
        options: [
          { id: "A", text: "Convert all data to CSV format for faster scanning" },
          { id: "B", text: "Use S3 Intelligent-Tiering for automatic optimization" },
          { id: "C", text: "Implement partition pruning and consider Z-ordering for frequently filtered columns" },
          { id: "D", text: "Copy recent data to a separate S3 bucket" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: CSV is less efficient than Parquet for analytical queries. Parquet's columnar format allows reading only needed columns.",
          B: "INCORRECT: S3 Intelligent-Tiering optimizes storage costs based on access patterns, not query performance. It doesn't help with faster data retrieval for analytics.",
          C: "CORRECT: Partition pruning ensures queries only scan relevant date partitions (last 30 days). Z-ordering (or data clustering) co-locates related data, reducing the amount of data scanned for common filter patterns. Both are standard data lake optimization techniques.",
          D: "INCORRECT: Duplicating data adds storage costs and synchronization complexity. Partition pruning achieves the same performance benefit without duplication."
        }
      },
      {
        id: 6,
        domain: "Data Engineering",
        question: "An ML pipeline reads training data from a PostgreSQL database. The database becomes slow during training due to heavy read operations. How can this be resolved while maintaining data freshness within 1 hour?",
        options: [
          { id: "A", text: "Increase PostgreSQL instance size" },
          { id: "B", text: "Create a read replica and point training jobs to the replica" },
          { id: "C", text: "Use AWS DMS to replicate data to S3, then train from S3" },
          { id: "D", text: "Cache all data in Redis before training" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Vertical scaling has limits and doesn't address the fundamental issue of mixing OLTP workloads with heavy analytical reads.",
          B: "INCORRECT: A read replica still uses the same PostgreSQL engine not optimized for large analytical scans. It helps but doesn't fully solve the problem for ML-scale reads.",
          C: "CORRECT: AWS DMS with CDC (Change Data Capture) can continuously replicate data to S3 with minimal lag. Training jobs read from S3 without impacting the production database. S3 is optimized for large sequential reads typical in ML training.",
          D: "INCORRECT: Caching entire training datasets in Redis is expensive and impractical for large datasets. Redis is for low-latency access to small amounts of data, not bulk training data."
        }
      },
      // Domain 2: Exploratory Data Analysis (7 questions)
      {
        id: 7,
        domain: "Exploratory Data Analysis",
        question: "A dataset contains timestamps showing when customers made purchases. The data scientist wants to engineer features that capture customer behavior patterns. Which feature engineering approach would provide the MOST predictive power?",
        options: [
          { id: "A", text: "Convert timestamps to Unix epoch integers" },
          { id: "B", text: "Extract cyclical features (hour, day of week, month) using sine/cosine encoding" },
          { id: "C", text: "Remove timestamps as they are unique identifiers" },
          { id: "D", text: "Bin timestamps into morning/afternoon/evening categories" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Unix timestamps as integers don't capture cyclical patterns. The model won't understand that hour 23 and hour 0 are close together.",
          B: "CORRECT: Sine/cosine encoding preserves the cyclical nature of time features. For example, hour 23 and hour 1 are encoded as close values, capturing the reality that late night and early morning shopping behaviors are similar. This applies to days of week, months, etc.",
          C: "INCORRECT: Timestamps contain valuable behavioral information about when customers shop. Removing them discards predictive signals.",
          D: "INCORRECT: Coarse binning loses information. Sine/cosine encoding preserves continuous relationships while maintaining cyclical properties."
        }
      },
      {
        id: 8,
        domain: "Exploratory Data Analysis",
        question: "A machine learning engineer discovers that a feature has values spanning several orders of magnitude (from 0.001 to 1,000,000). The feature contains zeros that cannot be removed. Which transformation is most appropriate?",
        options: [
          { id: "A", text: "Log transformation: log(x)" },
          { id: "B", text: "Log transformation with offset: log(x + 1)" },
          { id: "C", text: "Square root transformation" },
          { id: "D", text: "Reciprocal transformation: 1/x" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: log(0) is undefined (negative infinity). This transformation cannot handle the zeros in the data.",
          B: "CORRECT: Adding 1 before taking the log handles zeros (log(0+1) = log(1) = 0) while still compressing the range of large values. This is a standard technique for handling zero-inflated data with large ranges.",
          C: "INCORRECT: Square root helps with right-skewed data but doesn't compress the range as effectively as log for data spanning 6+ orders of magnitude.",
          D: "INCORRECT: Reciprocal transformation: 1/0 is undefined. This cannot handle zeros and inverts the scale in unintuitive ways."
        }
      },
      {
        id: 9,
        domain: "Exploratory Data Analysis",
        question: "A data scientist is analyzing customer segments and wants to visualize high-dimensional customer data (50 features) in 2D for exploration. The scientist needs to preserve local neighborhood relationships. Which technique is most appropriate?",
        options: [
          { id: "A", text: "Principal Component Analysis (PCA)" },
          { id: "B", text: "t-SNE (t-Distributed Stochastic Neighbor Embedding)" },
          { id: "C", text: "Linear Discriminant Analysis (LDA)" },
          { id: "D", text: "Factor Analysis" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: PCA preserves global variance and linear relationships but doesn't specifically optimize for preserving local neighborhoods. It may not reveal cluster structures well in 2D.",
          B: "CORRECT: t-SNE is specifically designed for visualization, optimizing to keep similar points close together and dissimilar points far apart in the low-dimensional space. It excels at revealing cluster structures and local relationships.",
          C: "INCORRECT: LDA is a supervised technique requiring labels. It finds projections that maximize class separation, not unsupervised exploration.",
          D: "INCORRECT: Factor Analysis identifies latent factors but doesn't specifically optimize for 2D visualization or neighborhood preservation."
        }
      },
      {
        id: 10,
        domain: "Exploratory Data Analysis",
        question: "A dataset has a numerical feature where values below the 5th percentile and above the 95th percentile are suspected measurement errors. The data scientist wants to handle these outliers while preserving the overall distribution. Which approach is most appropriate?",
        options: [
          { id: "A", text: "Delete all rows containing outliers" },
          { id: "B", text: "Replace outliers with the mean value" },
          { id: "C", text: "Winsorize the data by capping at the 5th and 95th percentiles" },
          { id: "D", text: "Apply log transformation to compress outliers" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Deleting 10% of data loses substantial information and may introduce bias if outliers are not random.",
          B: "INCORRECT: Replacing extreme values with the mean pulls them toward the center unnaturally and can distort the distribution shape.",
          C: "CORRECT: Winsorization caps extreme values at specified percentiles, limiting outlier impact while preserving the number of observations and general distribution shape. Values below 5th percentile become the 5th percentile value; above 95th become the 95th.",
          D: "INCORRECT: Log transformation changes the entire distribution, not just outliers. It's used for skewness, not targeted outlier handling."
        }
      },
      {
        id: 11,
        domain: "Exploratory Data Analysis",
        question: "A data scientist notices that a feature has a bimodal distribution with two distinct peaks. This feature represents customer age in a product used by both teenagers and parents. How should this be handled for a linear model?",
        options: [
          { id: "A", text: "Apply log transformation to normalize the distribution" },
          { id: "B", text: "Split into two binary indicator variables for each mode" },
          { id: "C", text: "Replace with the overall median value" },
          { id: "D", text: "Use standardization (z-score) to center the data" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Log transformation doesn't fix bimodal distributions. It helps with right-skewed data but won't merge or separate two distinct modes.",
          B: "CORRECT: Creating indicator variables (e.g., 'is_teenager', 'is_parent') captures the distinct segments. Linear models can then learn different coefficients for each group, properly modeling the non-linear relationship in the original feature.",
          C: "INCORRECT: Replacing with median loses all information about the bimodal structure, destroying the signal that two distinct customer segments exist.",
          D: "INCORRECT: Standardization centers and scales but doesn't change the bimodal shape. Linear models would still struggle with the non-linear relationship."
        }
      },
      {
        id: 12,
        domain: "Exploratory Data Analysis",
        question: "A fraud detection dataset has 200 features. A data scientist wants to identify which features are most important for predicting fraud BEFORE training a model. Which technique provides feature importance without requiring model training?",
        options: [
          { id: "A", text: "SHAP values" },
          { id: "B", text: "Permutation importance" },
          { id: "C", text: "Mutual information between each feature and target" },
          { id: "D", text: "Model coefficients from logistic regression" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: SHAP values require a trained model to explain. They measure how features contribute to a specific model's predictions.",
          B: "INCORRECT: Permutation importance also requires a trained model. It measures performance degradation when feature values are shuffled.",
          C: "CORRECT: Mutual information measures the statistical dependency between each feature and the target variable without training any model. It captures non-linear relationships and provides a model-agnostic importance score.",
          D: "INCORRECT: Obtaining coefficients requires training the logistic regression model first. This is post-training feature importance."
        }
      },
      {
        id: 13,
        domain: "Exploratory Data Analysis",
        question: "A text classification task involves documents with varying lengths (10 to 10,000 words). The data scientist wants to create fixed-length numerical representations for ML models. Which approach best handles variable document lengths?",
        options: [
          { id: "A", text: "Truncate all documents to 10 words" },
          { id: "B", text: "Pad all documents to 10,000 words with zeros" },
          { id: "C", text: "Use TF-IDF vectors with vocabulary-sized fixed dimensions" },
          { id: "D", text: "Count only the first sentence of each document" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Truncating to 10 words loses almost all information from longer documents. Many important terms would be cut off.",
          B: "INCORRECT: Padding to 10,000 words creates extremely sparse, high-dimensional vectors that waste memory and computation, especially for short documents.",
          C: "CORRECT: TF-IDF creates fixed-length vectors based on vocabulary size regardless of document length. Each document becomes a vector where each dimension represents a term's importance. Short and long documents are represented equally.",
          D: "INCORRECT: Using only the first sentence loses the majority of document content and context, severely limiting classification accuracy."
        }
      },
      // Domain 3: Modeling (11 questions)
      {
        id: 14,
        domain: "Modeling",
        question: "A model achieves 95% accuracy on both training and test sets but performs poorly in production. Investigation reveals the test set was created by random sampling from the same time period as training. What is the likely issue?",
        options: [
          { id: "A", text: "Underfitting due to insufficient model complexity" },
          { id: "B", text: "Temporal data leakage in the train/test split" },
          { id: "C", text: "Overfitting to the training data" },
          { id: "D", text: "Class imbalance in the dataset" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: 95% accuracy on both sets suggests the model has learned patterns well, not underfitting.",
          B: "CORRECT: Random sampling from the same time period causes temporal leakage—the model sees future information during training that wouldn't be available in production. Time-series data should use temporal splits (train on past, test on future) to properly evaluate production performance.",
          C: "INCORRECT: Similar training and test accuracy (95% vs 95%) indicates no overfitting. Overfitting shows high training accuracy with lower test accuracy.",
          D: "INCORRECT: Class imbalance would typically show up as inflated accuracy with poor minority class performance, not good test performance with poor production performance."
        }
      },
      {
        id: 15,
        domain: "Modeling",
        question: "A data scientist is building a model to predict rare equipment failures (0.01% of events). Standard classification algorithms predict all events as non-failures. Which approach would be MOST effective?",
        options: [
          { id: "A", text: "Increase the decision threshold from 0.5 to 0.9" },
          { id: "B", text: "Use anomaly detection algorithms instead of classification" },
          { id: "C", text: "Remove most non-failure examples to balance classes" },
          { id: "D", text: "Add more features to the model" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Increasing the threshold would make the model even more conservative, predicting failures even less often. The threshold should be lowered, not raised.",
          B: "CORRECT: At 0.01% positive rate, the problem is better framed as anomaly detection than binary classification. Algorithms like Isolation Forest or Random Cut Forest are designed to identify rare events without requiring balanced classes.",
          C: "INCORRECT: Removing 99.99% of data to balance classes would leave almost no training data and discard valuable information about normal operation patterns.",
          D: "INCORRECT: More features don't address the fundamental class imbalance issue. The model still won't learn to predict the minority class effectively."
        }
      },
      {
        id: 16,
        domain: "Modeling",
        question: "An image classification model trained on 224x224 images needs to classify images of varying sizes in production (from 100x100 to 4000x4000). What is the best approach?",
        options: [
          { id: "A", text: "Reject images that aren't exactly 224x224" },
          { id: "B", text: "Resize all input images to 224x224 during inference preprocessing" },
          { id: "C", text: "Retrain the model on all possible image sizes" },
          { id: "D", text: "Crop a 224x224 section from the center of each image" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Rejecting valid images is not practical for a production system that needs to handle varied inputs.",
          B: "CORRECT: Standard practice is to resize images to the model's expected input size during preprocessing. Modern interpolation methods (bilinear, bicubic) handle this well. The model's convolutional layers were trained to recognize patterns at 224x224 scale.",
          C: "INCORRECT: CNNs with fixed fully-connected layers require fixed input sizes. You cannot train a single model on arbitrary sizes without architectural changes.",
          D: "INCORRECT: Center cropping discards most of the image content, especially for large images. Important objects might be outside the crop region."
        }
      },
      {
        id: 17,
        domain: "Modeling",
        question: "A customer churn model needs to be interpretable for regulatory compliance. The model must explain why each customer was flagged as high churn risk. Which modeling approach satisfies this requirement?",
        options: [
          { id: "A", text: "Deep neural network with 10 hidden layers" },
          { id: "B", text: "Random Forest with 1000 trees" },
          { id: "C", text: "Gradient boosting with SHAP explanations" },
          { id: "D", text: "Support Vector Machine with RBF kernel" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Deep neural networks are black boxes. While techniques like LIME exist, they provide approximations, not exact explanations.",
          B: "INCORRECT: Random Forests with many trees are difficult to interpret at the individual prediction level. Feature importance is aggregate, not per-prediction.",
          C: "CORRECT: SHAP (SHapley Additive exPlanations) provides mathematically grounded, per-prediction explanations showing exactly how each feature contributed to that specific customer's churn score. This satisfies regulatory requirements for model interpretability.",
          D: "INCORRECT: SVMs with RBF kernels operate in transformed infinite-dimensional space, making individual predictions essentially unexplainable."
        }
      },
      {
        id: 18,
        domain: "Modeling",
        question: "A multi-label classification problem requires predicting multiple tags for each item (e.g., an article can be tagged as both 'politics' AND 'economy'). Which approach correctly handles multi-label classification?",
        options: [
          { id: "A", text: "Use softmax activation in the output layer" },
          { id: "B", text: "Use sigmoid activation with binary cross-entropy loss for each label" },
          { id: "C", text: "Combine all label combinations into a single multi-class problem" },
          { id: "D", text: "Train separate models and take the majority vote" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Softmax forces outputs to sum to 1, making labels mutually exclusive. It's for multi-class (one label only), not multi-label problems.",
          B: "CORRECT: Sigmoid activation outputs independent probabilities for each label (each can be 0-1 independently). Binary cross-entropy treats each label as a separate binary classification, allowing multiple labels to be predicted simultaneously.",
          C: "INCORRECT: With k labels, there are 2^k possible combinations. This explodes quickly and treats each combination as unrelated (e.g., 'politics+economy' shares nothing with 'politics').",
          D: "INCORRECT: Separate models don't capture label correlations (e.g., 'politics' and 'economy' often co-occur). It's also computationally inefficient."
        }
      },
      {
        id: 19,
        domain: "Modeling",
        question: "A regression model predicts house prices. The RMSE on test data is $50,000, but the model frequently predicts negative prices for small houses. How should this be addressed?",
        options: [
          { id: "A", text: "Remove small houses from the training data" },
          { id: "B", text: "Use a model that naturally predicts positive values (e.g., predict log-price)" },
          { id: "C", text: "Post-process predictions by replacing negatives with zero" },
          { id: "D", text: "Add a penalty term for negative predictions" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Removing data biases the model and doesn't solve the fundamental problem of the model being able to predict invalid values.",
          B: "CORRECT: Predicting log(price) and exponentiating ensures all predictions are positive. This is standard practice for price prediction. Alternatively, use models with appropriate link functions (e.g., Gamma regression) that naturally constrain outputs.",
          C: "INCORRECT: Replacing negatives with zero is a hacky fix that indicates the model isn't learning the true relationship. It also creates discontinuities.",
          D: "INCORRECT: Custom penalties complicate training and don't guarantee positive outputs. The log transformation is simpler and more effective."
        }
      },
      {
        id: 20,
        domain: "Modeling",
        question: "A SageMaker training job uses 4 ml.p3.16xlarge instances (each with 8 GPUs). Training is only using 1 GPU per instance. What is likely the issue?",
        options: [
          { id: "A", text: "The instance type doesn't support multiple GPUs" },
          { id: "B", text: "The training script isn't configured for distributed/multi-GPU training" },
          { id: "C", text: "SageMaker automatically limits GPU usage" },
          { id: "D", text: "The dataset is too small for multiple GPUs" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: ml.p3.16xlarge instances have 8 V100 GPUs and fully support multi-GPU workloads.",
          B: "CORRECT: Using multiple GPUs requires explicit configuration in the training script—either data parallelism (distributing batches) or model parallelism (distributing model layers). Without distributed training frameworks (Horovod, PyTorch DDP, etc.), only one GPU is used by default.",
          C: "INCORRECT: SageMaker doesn't artificially limit GPU usage. It provides the hardware; utilization depends on the training script.",
          D: "INCORRECT: Dataset size doesn't automatically determine GPU usage. Even small datasets can use multiple GPUs if the training script is configured for it."
        }
      },
      {
        id: 21,
        domain: "Modeling",
        question: "A model is trained on English text and needs to be adapted for French with limited French training data. Which transfer learning approach is most appropriate?",
        options: [
          { id: "A", text: "Train a new model from scratch on French data only" },
          { id: "B", text: "Fine-tune the pre-trained English model on French data" },
          { id: "C", text: "Use the English model directly without any adaptation" },
          { id: "D", text: "Translate all French data to English and use the original model" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Training from scratch with limited data would underperform. The model can't learn language fundamentals with small datasets.",
          B: "CORRECT: Fine-tuning transfers learned representations from the English model. Many linguistic concepts (syntax patterns, semantic relationships) transfer across languages. The model only needs to adapt vocabulary and language-specific patterns.",
          C: "INCORRECT: Direct use without adaptation would fail because the English model doesn't understand French vocabulary and grammar specifics.",
          D: "INCORRECT: Translation introduces errors and loses nuance. It's an indirect approach when direct fine-tuning is more effective."
        }
      },
      {
        id: 22,
        domain: "Modeling",
        question: "A reinforcement learning agent learns to play a game perfectly in simulation but fails completely when deployed to the real environment. What is this problem called?",
        options: [
          { id: "A", text: "Reward hacking" },
          { id: "B", text: "Catastrophic forgetting" },
          { id: "C", text: "Sim-to-real gap (domain shift)" },
          { id: "D", text: "Exploration-exploitation tradeoff" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Reward hacking is when an agent finds unexpected ways to maximize reward without solving the intended task. The scenario describes environment mismatch, not reward exploitation.",
          B: "INCORRECT: Catastrophic forgetting is when learning new tasks causes the model to forget previous tasks. This scenario is about simulation vs reality, not sequential learning.",
          C: "CORRECT: Sim-to-real gap occurs when policies learned in simulation don't transfer to real environments due to differences in physics, sensors, noise, and dynamics. Domain randomization and system identification are techniques to address this.",
          D: "INCORRECT: Exploration-exploitation is about balancing trying new actions vs using known good actions. It doesn't explain simulation-to-reality failure."
        }
      },
      {
        id: 23,
        domain: "Modeling",
        question: "A sequence-to-sequence model for machine translation produces repetitive outputs (e.g., 'the the the the...'). Which technique would MOST likely fix this?",
        options: [
          { id: "A", text: "Increase the model size" },
          { id: "B", text: "Use beam search with length normalization and repetition penalty" },
          { id: "C", text: "Train for more epochs" },
          { id: "D", text: "Use a higher learning rate" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Model size doesn't directly address the repetition problem, which is often a decoding issue rather than a capacity issue.",
          B: "CORRECT: Repetition in seq2seq models is typically a decoding problem. Beam search with repetition penalty explicitly discourages generating repeated tokens. Length normalization prevents the model from preferring shorter outputs that happen to repeat.",
          C: "INCORRECT: More training epochs might help if the model is undertrained, but repetition is usually a decoding-time issue, not a training issue.",
          D: "INCORRECT: Higher learning rate could destabilize training but doesn't specifically address repetition during generation."
        }
      },
      {
        id: 24,
        domain: "Modeling",
        question: "A classification model's ROC-AUC score is 0.95 but the precision-recall AUC is only 0.15. What does this indicate about the dataset?",
        options: [
          { id: "A", text: "The model is overfitting" },
          { id: "B", text: "The dataset has severe class imbalance" },
          { id: "C", text: "The features are not predictive" },
          { id: "D", text: "The model has high variance" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Overfitting would show in train/test performance gap, not in this metric discrepancy.",
          B: "CORRECT: ROC-AUC can appear high even with imbalanced data because it considers true negative rate. PR-AUC focuses on the positive class and is much more sensitive to imbalance. A high ROC-AUC with low PR-AUC is a classic signature of severe class imbalance.",
          C: "INCORRECT: If features weren't predictive, ROC-AUC would also be low (near 0.5).",
          D: "INCORRECT: High variance (overfitting) isn't specifically indicated by the ROC-AUC vs PR-AUC discrepancy."
        }
      },
      // Domain 4: ML Implementation and Operations (6 questions)
      {
        id: 25,
        domain: "ML Implementation and Operations",
        question: "A production ML model's accuracy dropped from 92% to 78% over two weeks. CloudWatch shows no infrastructure issues. What should be investigated first?",
        options: [
          { id: "A", text: "Check if the model was accidentally rolled back to an older version" },
          { id: "B", text: "Analyze input data distribution for drift using SageMaker Model Monitor" },
          { id: "C", text: "Increase the endpoint instance size" },
          { id: "D", text: "Retrain immediately with the same training data" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: While possible, gradual degradation over two weeks suggests drift rather than sudden version change. Version rollback would cause immediate, not gradual, performance change.",
          B: "CORRECT: Gradual performance degradation is typically caused by data drift—the input distribution changing over time. SageMaker Model Monitor tracks data quality and distribution changes, which should be the first investigation for gradual accuracy decline.",
          C: "INCORRECT: Instance size affects latency and throughput, not model accuracy. Infrastructure was already confirmed as not the issue.",
          D: "INCORRECT: Retraining on old data would reproduce the same model. The issue is that the production data has drifted from the training distribution."
        }
      },
      {
        id: 26,
        domain: "ML Implementation and Operations",
        question: "A company wants to deploy a new ML model version while ensuring zero downtime and the ability to quickly rollback. Which deployment strategy is most appropriate?",
        options: [
          { id: "A", text: "Delete the old endpoint and create a new one" },
          { id: "B", text: "Use SageMaker blue/green deployment with production variants" },
          { id: "C", text: "Deploy to a separate endpoint and update DNS" },
          { id: "D", text: "Stop the endpoint, update the model, and restart" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Deleting and recreating causes downtime and doesn't provide rollback capability.",
          B: "CORRECT: Blue/green deployment with production variants runs both versions simultaneously. Traffic can be shifted gradually (canary) or all at once. Rollback is instant by shifting traffic back to the old variant. Zero downtime guaranteed.",
          C: "INCORRECT: DNS changes have propagation delays and don't provide instant rollback. It's a valid strategy but not as clean as SageMaker's native blue/green.",
          D: "INCORRECT: Stopping the endpoint causes downtime, which violates the zero-downtime requirement."
        }
      },
      {
        id: 27,
        domain: "ML Implementation and Operations",
        question: "A SageMaker endpoint costs $500/day but only receives traffic for 4 hours during business hours. How can costs be optimized while maintaining real-time inference capability?",
        options: [
          { id: "A", text: "Use batch transform instead of real-time endpoints" },
          { id: "B", text: "Switch to SageMaker Serverless Inference" },
          { id: "C", text: "Reduce the instance size" },
          { id: "D", text: "Use spot instances for the endpoint" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Batch transform processes data in batches with higher latency, not suitable for real-time inference requirements.",
          B: "CORRECT: Serverless Inference scales to zero when not in use (20 hours/day) and only charges for actual inference time. For 4 hours of usage, costs would be dramatically reduced while maintaining real-time capability.",
          C: "INCORRECT: Reducing instance size still incurs 24/7 costs for 20 hours of idle time. It's less efficient than serverless for sporadic traffic.",
          D: "INCORRECT: SageMaker real-time endpoints don't support spot instances. Spot is available for training jobs, not inference endpoints."
        }
      },
      {
        id: 28,
        domain: "ML Implementation and Operations",
        question: "A model serving 1000 requests/second starts returning 5xx errors under load. The endpoint uses auto-scaling but new instances take 5 minutes to become ready. What is the best solution?",
        options: [
          { id: "A", text: "Increase the auto-scaling cooldown period" },
          { id: "B", text: "Use provisioned concurrency or pre-warming to reduce cold start time" },
          { id: "C", text: "Disable auto-scaling and use a fixed large fleet" },
          { id: "D", text: "Reduce the model size to speed up loading" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Longer cooldown would slow down scaling response, making the problem worse during traffic spikes.",
          B: "CORRECT: Pre-warming (keeping warm instances ready) or provisioned capacity ensures instances are ready before traffic spikes. This addresses the 5-minute cold start that causes errors during scaling events.",
          C: "INCORRECT: A fixed fleet large enough for peak load would be very expensive during off-peak times. Auto-scaling is still valuable; the issue is cold start time.",
          D: "INCORRECT: While smaller models load faster, 5 minutes is typically due to container startup and model loading. Optimization helps but pre-warming is more effective."
        }
      },
      {
        id: 29,
        domain: "ML Implementation and Operations",
        question: "An ML team has multiple data scientists training models with different frameworks (TensorFlow, PyTorch, XGBoost). They need to standardize model packaging for deployment. Which approach provides framework-agnostic model packaging?",
        options: [
          { id: "A", text: "Require all data scientists to use TensorFlow only" },
          { id: "B", text: "Use MLflow for experiment tracking and model packaging" },
          { id: "C", text: "Save all models as pickle files" },
          { id: "D", text: "Convert all models to PMML format" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Forcing a single framework limits flexibility and may not be optimal for all use cases. Different problems benefit from different frameworks.",
          B: "CORRECT: MLflow provides a standard Model format that packages models from any framework with their dependencies, requirements, and serving signatures. It integrates with SageMaker for deployment, providing framework-agnostic packaging.",
          C: "INCORRECT: Pickle files are Python-specific, not portable, and have security concerns. They also don't capture dependencies or serving requirements.",
          D: "INCORRECT: PMML has limited support for modern deep learning models and doesn't capture all model types equally well."
        }
      },
      {
        id: 30,
        domain: "ML Implementation and Operations",
        question: "A model running on a SageMaker endpoint occasionally returns predictions that don't match expected business rules (e.g., negative prices, impossible dates). How should this be addressed in production?",
        options: [
          { id: "A", text: "Retrain the model to learn the business rules" },
          { id: "B", text: "Implement inference pipeline with a post-processing step for validation" },
          { id: "C", text: "Add more training data with valid examples" },
          { id: "D", text: "Increase model complexity to capture constraints" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Models don't reliably enforce hard constraints. Even with training, statistical models can produce invalid outputs.",
          B: "CORRECT: SageMaker inference pipelines allow chaining preprocessing and postprocessing containers. A postprocessing step can validate outputs against business rules, clamp values to valid ranges, or flag anomalous predictions—guaranteeing valid outputs.",
          C: "INCORRECT: More training data doesn't guarantee the model will never produce invalid outputs. Statistical models can still extrapolate to invalid regions.",
          D: "INCORRECT: Model complexity doesn't help enforce hard business rules. This is a constraint satisfaction problem, not a learning capacity problem."
        }
      }
    ]
  },
  test3: {
    name: "Quick Quiz 3",
    description: "15-question rapid assessment across all domains",
    questions: [
      {
        id: 1,
        domain: "Data Engineering",
        question: "A company wants to build a real-time fraud detection system. Transaction data arrives at 10,000 events per second. The ML model needs to score transactions within 100ms. Which architecture is most appropriate?",
        options: [
          { id: "A", text: "Kinesis Data Streams → Lambda → SageMaker Batch Transform" },
          { id: "B", text: "Kinesis Data Streams → Lambda → SageMaker Real-time Endpoint" },
          { id: "C", text: "S3 → AWS Glue → SageMaker Processing" },
          { id: "D", text: "SQS → EC2 → Custom inference code" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Batch Transform is for offline batch processing, not real-time inference. It cannot meet the 100ms latency requirement.",
          B: "CORRECT: Kinesis Data Streams handles high-throughput streaming ingestion. Lambda processes events and invokes a SageMaker real-time endpoint for low-latency inference, meeting the 100ms requirement.",
          C: "INCORRECT: This is a batch processing architecture. Glue and Processing jobs are for batch workloads, not real-time scoring.",
          D: "INCORRECT: SQS adds latency and doesn't guarantee ordering. Custom EC2 inference lacks the managed scaling and reliability of SageMaker endpoints."
        }
      },
      {
        id: 2,
        domain: "Data Engineering",
        question: "An ML team needs to store features for both real-time inference (single-digit millisecond latency) and batch training jobs. Which AWS service provides this dual capability?",
        options: [
          { id: "A", text: "Amazon ElastiCache" },
          { id: "B", text: "Amazon DynamoDB with DAX" },
          { id: "C", text: "Amazon SageMaker Feature Store" },
          { id: "D", text: "Amazon Redshift with materialized views" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: ElastiCache provides low-latency caching but doesn't integrate with ML training workflows or provide feature versioning.",
          B: "INCORRECT: DynamoDB with DAX offers low latency but lacks ML-specific features like feature versioning, lineage tracking, and S3 integration for training.",
          C: "CORRECT: SageMaker Feature Store has both an Online Store (low-latency retrieval for inference) and Offline Store (S3-based for batch training). It automatically syncs features between stores.",
          D: "INCORRECT: Redshift is a data warehouse optimized for analytics, not real-time feature serving with single-digit ms latency."
        }
      },
      {
        id: 3,
        domain: "Data Engineering",
        question: "A data scientist needs to automatically discover schema information from CSV and JSON files stored across multiple S3 buckets and make them queryable. Which service combination is most efficient?",
        options: [
          { id: "A", text: "AWS Lambda to parse files and store schema in DynamoDB" },
          { id: "B", text: "AWS Glue Crawlers to populate the Glue Data Catalog, query with Athena" },
          { id: "C", text: "Amazon EMR with Hive metastore" },
          { id: "D", text: "Amazon Redshift Spectrum with manual table definitions" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Building custom schema discovery with Lambda requires significant development effort and maintenance.",
          B: "CORRECT: Glue Crawlers automatically discover schema and populate the Data Catalog. Athena can then query the data directly in S3 using the discovered schemas—fully serverless and minimal setup.",
          C: "INCORRECT: EMR requires cluster management and Hive metastore setup. More operational overhead than the serverless Glue/Athena combination.",
          D: "INCORRECT: Redshift Spectrum requires manual table definitions, which doesn't meet the automatic schema discovery requirement."
        }
      },
      {
        id: 4,
        domain: "Exploratory Data Analysis",
        question: "A dataset has a feature with values ranging from 0 to 1,000,000 with a heavy right skew. Most values are below 1,000 but some outliers reach millions. Which transformation is most appropriate before using this feature in a linear regression model?",
        options: [
          { id: "A", text: "Min-Max scaling to [0,1]" },
          { id: "B", text: "Z-score standardization" },
          { id: "C", text: "Log transformation" },
          { id: "D", text: "One-hot encoding" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Min-Max scaling preserves the skewed distribution. Outliers would compress most values near 0.",
          B: "INCORRECT: Z-score standardization also preserves the skewed distribution. The outliers would still have extreme z-scores.",
          C: "CORRECT: Log transformation compresses large values and spreads small values, effectively reducing right skewness. This makes the distribution more normal-like, which is beneficial for linear models.",
          D: "INCORRECT: One-hot encoding is for categorical variables, not continuous numerical features."
        }
      },
      {
        id: 5,
        domain: "Exploratory Data Analysis",
        question: "A classification dataset has 95% negative examples and 5% positive examples. A model achieves 95% accuracy. What can be concluded?",
        options: [
          { id: "A", text: "The model is performing excellently" },
          { id: "B", text: "The model may be predicting all negative, and accuracy is misleading" },
          { id: "C", text: "The model needs more training epochs" },
          { id: "D", text: "The dataset needs more features" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: 95% accuracy sounds good but equals the majority class proportion, suggesting the model may not be learning.",
          B: "CORRECT: With 95% negative examples, a model predicting all negative achieves 95% accuracy while being completely useless for the positive class. Use precision, recall, F1, or AUC instead.",
          C: "INCORRECT: More epochs won't help if the model is already converged to predicting the majority class.",
          D: "INCORRECT: More features won't address the class imbalance issue causing the misleading accuracy metric."
        }
      },
      {
        id: 6,
        domain: "Exploratory Data Analysis",
        question: "A text classification model needs to convert documents into numerical features. The goal is to weight words that are distinctive to specific documents higher than common words appearing everywhere. Which technique should be used?",
        options: [
          { id: "A", text: "Bag of Words with raw counts" },
          { id: "B", text: "TF-IDF (Term Frequency-Inverse Document Frequency)" },
          { id: "C", text: "One-hot encoding of words" },
          { id: "D", text: "Label encoding of words" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Raw word counts don't penalize common words. 'the' and 'a' would have high counts but low discriminative power.",
          B: "CORRECT: TF-IDF multiplies term frequency by inverse document frequency, which downweights words appearing in many documents (common words) and upweights rare, distinctive words.",
          C: "INCORRECT: One-hot encoding creates sparse vectors but doesn't weight by importance or frequency.",
          D: "INCORRECT: Label encoding assigns arbitrary integers to words, implying false ordinal relationships."
        }
      },
      {
        id: 7,
        domain: "Exploratory Data Analysis",
        question: "A dataset contains a 'hour_of_day' feature (0-23) for predicting customer behavior. How should this cyclical feature be encoded for a neural network?",
        options: [
          { id: "A", text: "Use the raw integer values 0-23" },
          { id: "B", text: "One-hot encode into 24 binary features" },
          { id: "C", text: "Use sine and cosine transformations" },
          { id: "D", text: "Normalize to range [0,1]" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Raw integers imply hour 23 is far from hour 0, when they're actually adjacent (11pm and midnight).",
          B: "INCORRECT: One-hot encoding doesn't capture that hour 23 and hour 0 are close together.",
          C: "CORRECT: sin(2π*hour/24) and cos(2π*hour/24) create a circular representation where hour 23 and hour 0 are adjacent, properly encoding the cyclical nature.",
          D: "INCORRECT: Normalizing to [0,1] still implies hour 23 (≈1) is far from hour 0 (0), missing the cyclical relationship."
        }
      },
      {
        id: 8,
        domain: "Modeling",
        question: "A SageMaker XGBoost model shows low training error but high validation error. Which hyperparameter change would most likely help?",
        options: [
          { id: "A", text: "Increase max_depth" },
          { id: "B", text: "Increase num_round" },
          { id: "C", text: "Decrease max_depth and increase min_child_weight" },
          { id: "D", text: "Increase eta (learning rate)" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Increasing max_depth allows deeper trees, which would increase overfitting (already the problem).",
          B: "INCORRECT: More boosting rounds with an already overfitting model would make overfitting worse.",
          C: "CORRECT: This is classic overfitting. Decreasing max_depth creates shallower trees. Increasing min_child_weight requires more samples per leaf, both reducing model complexity.",
          D: "INCORRECT: Higher learning rate can lead to less stable training but doesn't directly address overfitting."
        }
      },
      {
        id: 9,
        domain: "Modeling",
        question: "A multi-class classification problem has 5 mutually exclusive classes. Which output layer activation and loss function should be used in a neural network?",
        options: [
          { id: "A", text: "Sigmoid activation with binary cross-entropy loss" },
          { id: "B", text: "Softmax activation with categorical cross-entropy loss" },
          { id: "C", text: "ReLU activation with mean squared error loss" },
          { id: "D", text: "Tanh activation with hinge loss" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Sigmoid with binary cross-entropy is for binary classification or multi-label (non-mutually exclusive) problems.",
          B: "CORRECT: Softmax outputs probabilities summing to 1, perfect for mutually exclusive classes. Categorical cross-entropy is the standard loss for multi-class classification.",
          C: "INCORRECT: ReLU is for hidden layers. MSE is for regression, not classification.",
          D: "INCORRECT: Tanh outputs [-1,1], not probabilities. Hinge loss is typically used with SVMs, not standard neural network classification."
        }
      },
      {
        id: 10,
        domain: "Modeling",
        question: "A company wants to build a recommendation system but has no historical user-item interaction data. They only have item metadata (descriptions, categories). Which approach is most suitable?",
        options: [
          { id: "A", text: "Collaborative filtering" },
          { id: "B", text: "Matrix factorization" },
          { id: "C", text: "Content-based filtering" },
          { id: "D", text: "User-based nearest neighbors" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Collaborative filtering requires user-item interaction data (ratings, purchases) which doesn't exist.",
          B: "INCORRECT: Matrix factorization is a collaborative filtering technique that needs a user-item interaction matrix.",
          C: "CORRECT: Content-based filtering uses item features/metadata to recommend similar items. It works without user interaction history—solving the cold start problem.",
          D: "INCORRECT: User-based methods require historical user behavior data to find similar users."
        }
      },
      {
        id: 11,
        domain: "Modeling",
        question: "An ML specialist needs to forecast sales for 1000 products, each with its own time series. The products have similar seasonal patterns. Which SageMaker algorithm is best suited?",
        options: [
          { id: "A", text: "SageMaker Linear Learner" },
          { id: "B", text: "SageMaker XGBoost" },
          { id: "C", text: "SageMaker DeepAR" },
          { id: "D", text: "SageMaker K-Means" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Linear Learner is for regression/classification on tabular data, not specialized for time series forecasting.",
          B: "INCORRECT: XGBoost can do time series with feature engineering, but isn't designed to leverage patterns across multiple related series.",
          C: "CORRECT: DeepAR is specifically designed for forecasting multiple related time series. It learns patterns across all series and handles seasonality, making it ideal for forecasting many products together.",
          D: "INCORRECT: K-Means is for clustering, not time series forecasting."
        }
      },
      {
        id: 12,
        domain: "Modeling",
        question: "A model needs to classify images of handwritten digits (0-9). Which SageMaker built-in algorithm or approach is most appropriate?",
        options: [
          { id: "A", text: "SageMaker BlazingText" },
          { id: "B", text: "SageMaker Image Classification algorithm" },
          { id: "C", text: "SageMaker Random Cut Forest" },
          { id: "D", text: "SageMaker Factorization Machines" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: BlazingText is for text classification and Word2Vec embeddings, not image classification.",
          B: "CORRECT: SageMaker Image Classification uses deep learning (ResNet) to classify images into categories. Perfect for digit recognition with 10 classes.",
          C: "INCORRECT: Random Cut Forest is for anomaly detection, not image classification.",
          D: "INCORRECT: Factorization Machines are for recommendation systems and sparse data, not image classification."
        }
      },
      {
        id: 13,
        domain: "ML Implementation and Operations",
        question: "A company wants to deploy a model that receives unpredictable traffic—sometimes zero requests per hour, sometimes thousands. Cost optimization is critical. Which SageMaker deployment option is best?",
        options: [
          { id: "A", text: "Real-time endpoint with auto-scaling" },
          { id: "B", text: "Serverless inference" },
          { id: "C", text: "Batch transform" },
          { id: "D", text: "Asynchronous inference" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Real-time endpoints have a minimum instance running 24/7. Even with auto-scaling, you pay for idle time during zero-traffic periods.",
          B: "CORRECT: Serverless inference scales to zero during idle periods—you only pay for actual inference requests. Ideal for unpredictable, sparse traffic patterns.",
          C: "INCORRECT: Batch transform is for processing large batches, not on-demand inference requests.",
          D: "INCORRECT: Asynchronous inference is for large payloads with long processing times, not cost optimization for variable traffic."
        }
      },
      {
        id: 14,
        domain: "ML Implementation and Operations",
        question: "A deployed model's accuracy has degraded over 6 months. Investigation shows that customer behavior patterns have shifted since training. What is this phenomenon called and how should it be addressed?",
        options: [
          { id: "A", text: "Overfitting - retrain with regularization" },
          { id: "B", text: "Concept drift - retrain on recent data" },
          { id: "C", text: "Data leakage - fix feature engineering" },
          { id: "D", text: "Underfitting - increase model complexity" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Overfitting is high training / low validation performance during development, not degradation over time in production.",
          B: "CORRECT: Concept drift occurs when the statistical properties of the target variable change over time. The model learned old patterns that no longer apply. Retraining on recent data updates the model to current patterns.",
          C: "INCORRECT: Data leakage is using future information during training, causing overly optimistic evaluation—not gradual degradation.",
          D: "INCORRECT: Underfitting is poor performance from the start due to insufficient model capacity, not degradation over time."
        }
      },
      {
        id: 15,
        domain: "ML Implementation and Operations",
        question: "A financial services company needs to explain why their ML model denied a loan application to comply with regulations. Which SageMaker capability provides this?",
        options: [
          { id: "A", text: "SageMaker Debugger" },
          { id: "B", text: "SageMaker Model Monitor" },
          { id: "C", text: "SageMaker Clarify with SHAP values" },
          { id: "D", text: "SageMaker Experiments" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Debugger analyzes training jobs to find issues like vanishing gradients. It doesn't explain individual predictions.",
          B: "INCORRECT: Model Monitor tracks data quality and model drift over time. It monitors aggregate behavior, not individual prediction explanations.",
          C: "CORRECT: SageMaker Clarify uses SHAP (SHapley Additive exPlanations) to provide feature attributions showing which factors contributed to each prediction and by how much—essential for regulatory explainability.",
          D: "INCORRECT: Experiments tracks training runs and hyperparameters for comparison. It doesn't explain predictions."
        }
      }
    ]
  },
  test4: {
    name: "Quick Quiz 4",
    description: "15-question assessment focusing on practical scenarios",
    questions: [
      {
        id: 1,
        domain: "Data Engineering",
        question: "A company processes daily batch jobs that transform raw data into ML features. Sometimes jobs fail midway and need to restart. How can AWS Glue prevent reprocessing already-processed data?",
        options: [
          { id: "A", text: "Use AWS Glue job bookmarks" },
          { id: "B", text: "Implement custom checkpointing in the ETL script" },
          { id: "C", text: "Store processed file names in DynamoDB" },
          { id: "D", text: "Use S3 versioning to track processed files" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: Glue job bookmarks automatically track processed data. When a job restarts, it resumes from where it left off, avoiding reprocessing. This is a built-in Glue feature requiring minimal configuration.",
          B: "INCORRECT: Custom checkpointing works but requires significant development effort. Glue bookmarks provide this functionality out of the box.",
          C: "INCORRECT: Tracking in DynamoDB requires custom code and maintenance. Glue bookmarks are simpler.",
          D: "INCORRECT: S3 versioning tracks file versions but doesn't help Glue know which files were already processed."
        }
      },
      {
        id: 2,
        domain: "Data Engineering",
        question: "An ML pipeline needs to process video files to extract frames for computer vision training. The processing is compute-intensive and benefits from GPU acceleration. Which AWS service is most appropriate?",
        options: [
          { id: "A", text: "AWS Lambda with increased memory" },
          { id: "B", text: "AWS Glue with G.2X workers" },
          { id: "C", text: "AWS Batch with GPU instances" },
          { id: "D", text: "Amazon EMR with CPU instances" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Lambda doesn't support GPU instances and has a 15-minute timeout, insufficient for video processing.",
          B: "INCORRECT: Glue workers don't have GPU options. G.2X are larger memory workers, not GPU-enabled.",
          C: "CORRECT: AWS Batch supports GPU instances (P and G instance families), automatic job scheduling, and is designed for compute-intensive batch workloads like video processing.",
          D: "INCORRECT: EMR can work but doesn't natively support GPU acceleration as easily as Batch for this use case."
        }
      },
      {
        id: 3,
        domain: "Data Engineering",
        question: "A data scientist needs to detect and mask PII (names, addresses, SSNs) in a large dataset before ML training. Which AWS service provides automated PII detection and transformation?",
        options: [
          { id: "A", text: "Amazon Macie" },
          { id: "B", text: "AWS Glue with Sensitive Data Detection transform" },
          { id: "C", text: "Amazon Comprehend" },
          { id: "D", text: "AWS Lambda with custom regex" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Macie discovers PII in S3 for security monitoring but doesn't transform/mask data for ML pipelines.",
          B: "CORRECT: AWS Glue has a built-in Sensitive Data Detection transform that automatically identifies and masks PII types (names, SSNs, addresses) as part of ETL jobs—ideal for ML data preparation.",
          C: "INCORRECT: Comprehend detects PII in text via API but doesn't provide batch transformation capabilities for large datasets.",
          D: "INCORRECT: Custom regex requires significant development and may miss PII patterns. Glue's built-in transform is more comprehensive."
        }
      },
      {
        id: 4,
        domain: "Exploratory Data Analysis",
        question: "A dataset has missing values in 3% of records for a critical feature. The missing values appear random (MCAR). The dataset is large (10 million records). What is the best approach?",
        options: [
          { id: "A", text: "Delete all records with missing values" },
          { id: "B", text: "Use KNN imputation" },
          { id: "C", text: "Use mean/median imputation" },
          { id: "D", text: "Replace with zero" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: With only 3% missing (MCAR) and 10 million records, deleting 300K records leaves 9.7 million—more than sufficient. This is the simplest approach that avoids introducing imputation bias.",
          B: "INCORRECT: KNN imputation is computationally expensive on 10 million records and unnecessary when deletion is viable.",
          C: "INCORRECT: While valid, it introduces artificial values. With a large dataset and small missing percentage, deletion is cleaner.",
          D: "INCORRECT: Replacing with zero introduces bias unless zero is a meaningful value for the feature."
        }
      },
      {
        id: 5,
        domain: "Exploratory Data Analysis",
        question: "A categorical feature 'country' has 200 unique values. The ML model is tree-based (XGBoost). What is the most appropriate encoding method?",
        options: [
          { id: "A", text: "One-hot encoding" },
          { id: "B", text: "Label encoding" },
          { id: "C", text: "Target encoding (mean encoding)" },
          { id: "D", text: "Binary encoding" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: One-hot encoding creates 200 new features, leading to sparse data and slower training. Not ideal for high cardinality.",
          B: "INCORRECT: Label encoding assigns arbitrary integers, implying false ordinal relationships between countries.",
          C: "CORRECT: Target encoding replaces each category with the mean target value. It handles high cardinality well and works especially well with tree-based models. Use with cross-validation to prevent target leakage.",
          D: "INCORRECT: Binary encoding creates log2(200) ≈ 8 features but loses semantic meaning of categories."
        }
      },
      {
        id: 6,
        domain: "Exploratory Data Analysis",
        question: "Two features in a dataset have a Pearson correlation of 0.95. The model is logistic regression. What problem might this cause and how should it be addressed?",
        options: [
          { id: "A", text: "Overfitting - add dropout" },
          { id: "B", text: "Multicollinearity - remove one feature or use regularization" },
          { id: "C", text: "Underfitting - add polynomial features" },
          { id: "D", text: "Class imbalance - use SMOTE" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Dropout is for neural networks. High correlation between features is multicollinearity, not overfitting.",
          B: "CORRECT: Multicollinearity (highly correlated features) causes unstable coefficient estimates in linear models. Solutions include removing one redundant feature, using PCA, or applying L2 regularization.",
          C: "INCORRECT: Adding polynomial features would increase multicollinearity, not solve it.",
          D: "INCORRECT: Class imbalance is about target variable distribution, unrelated to feature correlation."
        }
      },
      {
        id: 7,
        domain: "Exploratory Data Analysis",
        question: "A fraud detection dataset has 0.1% fraud cases. Which technique would help the model learn to detect fraud better?",
        options: [
          { id: "A", text: "Increase the learning rate" },
          { id: "B", text: "Use SMOTE to generate synthetic fraud samples" },
          { id: "C", text: "Add more features" },
          { id: "D", text: "Reduce model complexity" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Learning rate affects training speed/stability but doesn't address class imbalance.",
          B: "CORRECT: SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic samples for the minority class by interpolating between existing fraud cases, helping the model learn fraud patterns.",
          C: "INCORRECT: More features don't address the class imbalance problem where the model sees very few fraud examples.",
          D: "INCORRECT: Model complexity is unrelated to class imbalance. The issue is insufficient minority class examples."
        }
      },
      {
        id: 8,
        domain: "Modeling",
        question: "A training job shows both training loss and validation loss decreasing together, but validation loss is consistently higher. After 100 epochs, both losses are still decreasing. What action should be taken?",
        options: [
          { id: "A", text: "Stop training - the model is overfitting" },
          { id: "B", text: "Continue training - the model is still learning" },
          { id: "C", text: "Reduce model complexity immediately" },
          { id: "D", text: "Add more regularization immediately" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Overfitting shows validation loss increasing while training loss decreases. Both decreasing indicates healthy learning.",
          B: "CORRECT: Both losses decreasing means the model is still learning and generalizing. The gap between them is normal. Continue training until validation loss plateaus or starts increasing.",
          C: "INCORRECT: No need to reduce complexity when the model is still improving on validation data.",
          D: "INCORRECT: Regularization combats overfitting, but this pattern shows healthy learning, not overfitting."
        }
      },
      {
        id: 9,
        domain: "Modeling",
        question: "An NLP task requires classifying customer support tickets into 50 categories. Limited labeled data is available (500 examples total). Which approach is most likely to succeed?",
        options: [
          { id: "A", text: "Train a custom LSTM from scratch" },
          { id: "B", text: "Use transfer learning with a pre-trained BERT model" },
          { id: "C", text: "Use bag of words with logistic regression" },
          { id: "D", text: "Train a custom transformer from scratch" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Training LSTM from scratch with only 500 examples across 50 classes (10 per class average) would severely overfit.",
          B: "CORRECT: Transfer learning with pre-trained BERT leverages knowledge learned from massive text corpora. Fine-tuning on limited data often achieves strong results because the model already understands language.",
          C: "INCORRECT: Bag of words loses word order and semantic meaning. With 50 classes and limited data, it would struggle.",
          D: "INCORRECT: Transformers have millions of parameters. Training from scratch requires massive data, not 500 examples."
        }
      },
      {
        id: 10,
        domain: "Modeling",
        question: "A Random Cut Forest model in SageMaker assigns an anomaly score of 3.5 to a data point. What does this indicate?",
        options: [
          { id: "A", text: "The point is definitely an anomaly" },
          { id: "B", text: "The point has 3.5 standard deviations from the mean" },
          { id: "C", text: "The point requires fewer cuts to isolate, suggesting anomaly" },
          { id: "D", text: "The point is 3.5 times more likely than average" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Anomaly scores require threshold interpretation. A score alone doesn't determine definite anomaly status.",
          B: "INCORRECT: RCF scores aren't standard deviations. They're based on tree depth required to isolate points.",
          C: "CORRECT: Random Cut Forest scores reflect how easily a point can be isolated. Anomalies are isolated with fewer random cuts (shorter path = higher score). Score 3.5 means relatively easy to isolate.",
          D: "INCORRECT: The score isn't a likelihood ratio. It's based on isolation path length."
        }
      },
      {
        id: 11,
        domain: "Modeling",
        question: "A binary classification model for medical diagnosis must catch 95% of positive cases (disease present) even if it means more false positives. Which metric should be optimized?",
        options: [
          { id: "A", text: "Precision" },
          { id: "B", text: "Recall (Sensitivity)" },
          { id: "C", text: "Specificity" },
          { id: "D", text: "Accuracy" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Precision measures what fraction of positive predictions are correct. Optimizing precision might miss disease cases.",
          B: "CORRECT: Recall = TP/(TP+FN) measures what fraction of actual positives are correctly identified. 95% recall means catching 95% of disease cases, which is the requirement.",
          C: "INCORRECT: Specificity measures true negative rate. Optimizing specificity would reduce false positives but might miss disease cases.",
          D: "INCORRECT: Accuracy doesn't distinguish between types of errors. Missing a disease case and false alarm are weighted equally."
        }
      },
      {
        id: 12,
        domain: "Modeling",
        question: "A SageMaker training job needs to use a custom deep learning framework not available in built-in containers. What is the recommended approach?",
        options: [
          { id: "A", text: "Install the framework using pip in the training script" },
          { id: "B", text: "Use SageMaker Bring Your Own Container (BYOC)" },
          { id: "C", text: "Use AWS Lambda for training" },
          { id: "D", text: "Request AWS to add the framework to SageMaker" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Installing frameworks at runtime adds startup latency and may fail due to dependencies or compilation requirements.",
          B: "CORRECT: BYOC allows you to create a Docker container with any framework, dependencies, and configurations. SageMaker runs your container for training with full control over the environment.",
          C: "INCORRECT: Lambda has limited compute, memory, and execution time. It's not designed for ML training.",
          D: "INCORRECT: Waiting for AWS to add frameworks isn't practical. BYOC provides immediate flexibility."
        }
      },
      {
        id: 13,
        domain: "ML Implementation and Operations",
        question: "A SageMaker model needs to be deployed to edge devices with limited compute power. The model was trained with TensorFlow. Which service optimizes the model for edge deployment?",
        options: [
          { id: "A", text: "SageMaker Model Monitor" },
          { id: "B", text: "SageMaker Neo" },
          { id: "C", text: "SageMaker Clarify" },
          { id: "D", text: "SageMaker Debugger" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Model Monitor tracks production model performance, it doesn't optimize models for deployment.",
          B: "CORRECT: SageMaker Neo compiles models for specific target hardware (edge devices, specific CPUs/GPUs). It optimizes model size and inference speed for resource-constrained environments.",
          C: "INCORRECT: Clarify provides bias detection and explainability, not model optimization.",
          D: "INCORRECT: Debugger analyzes training jobs for issues, it doesn't optimize for edge deployment."
        }
      },
      {
        id: 14,
        domain: "ML Implementation and Operations",
        question: "A company wants to deploy a new model version with minimal risk. They want to send 10% of traffic to the new version while monitoring for errors before full rollout. Which deployment strategy should they use?",
        options: [
          { id: "A", text: "Blue/Green deployment with instant cutover" },
          { id: "B", text: "Canary deployment with traffic shifting" },
          { id: "C", text: "Rolling deployment" },
          { id: "D", text: "Recreate deployment" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Blue/Green with instant cutover sends 100% traffic to the new version immediately—no gradual testing.",
          B: "CORRECT: Canary deployment routes a small percentage (10%) of traffic to the new version. If metrics look good, gradually increase. This minimizes risk by limiting blast radius.",
          C: "INCORRECT: Rolling deployment replaces instances gradually but doesn't provide precise traffic percentage control.",
          D: "INCORRECT: Recreate deployment stops old version completely before starting new—causes downtime and no gradual testing."
        }
      },
      {
        id: 15,
        domain: "ML Implementation and Operations",
        question: "A SageMaker training job needs to process a 500GB dataset. The team wants to minimize costs while maintaining reasonable training time. Which approach is recommended?",
        options: [
          { id: "A", text: "Use the largest available instance type" },
          { id: "B", text: "Use Spot instances with checkpointing" },
          { id: "C", text: "Use multiple on-demand instances" },
          { id: "D", text: "Use Serverless inference" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Largest instance types are expensive. Using them doesn't optimize cost.",
          B: "CORRECT: Spot instances offer up to 90% cost savings. Checkpointing saves training progress, so if the Spot instance is interrupted, training resumes from the checkpoint rather than starting over.",
          C: "INCORRECT: Multiple on-demand instances are more expensive than Spot instances.",
          D: "INCORRECT: Serverless inference is for inference/prediction, not training. Training requires dedicated compute instances."
        }
      }
    ]
  },
  test5: {
    name: "Full Practice Exam",
    description: "65-question exam simulation matching real exam format (180 min recommended)",
    questions: [
      // Domain 1: Data Engineering (13 questions - 20%)
      {
        id: 1,
        domain: "Data Engineering",
        question: "A company needs to collect clickstream data from their website in real-time and store it for ML model training. The data volume varies significantly throughout the day. Which architecture provides the most cost-effective and scalable solution?",
        options: [
          { id: "A", text: "Amazon API Gateway → AWS Lambda → Amazon S3" },
          { id: "B", text: "Amazon Kinesis Data Streams → Amazon Kinesis Data Firehose → Amazon S3" },
          { id: "C", text: "Amazon SQS → AWS Lambda → Amazon RDS" },
          { id: "D", text: "AWS Direct Connect → Amazon EC2 Auto Scaling → Amazon EBS" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: API Gateway with Lambda works for moderate traffic but has higher latency and cost at scale compared to Kinesis for streaming data.",
          B: "CORRECT: Kinesis Data Streams handles variable throughput with on-demand capacity. Firehose automatically batches and delivers to S3 with near real-time latency. This is the standard architecture for streaming ingestion to data lakes.",
          C: "INCORRECT: SQS with RDS is not designed for high-volume streaming data. RDS is not optimal for storing raw ML training data.",
          D: "INCORRECT: Direct Connect is for dedicated network connections, not data ingestion. EC2 requires manual scaling configuration."
        }
      },
      {
        id: 2,
        domain: "Data Engineering",
        question: "A data engineer needs to transform CSV files stored in S3 into Parquet format for more efficient ML training. The transformation should be serverless and run on a schedule. Which solution requires the least operational overhead?",
        options: [
          { id: "A", text: "Amazon EMR with scheduled Spark jobs" },
          { id: "B", text: "AWS Glue ETL jobs with scheduled triggers" },
          { id: "C", text: "AWS Lambda functions triggered by S3 events" },
          { id: "D", text: "Amazon EC2 instances with cron jobs" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: EMR requires cluster management—provisioning, scaling, termination. Not fully serverless.",
          B: "CORRECT: AWS Glue is serverless ETL. Jobs can be scheduled with triggers, automatically handle Parquet conversion, and require no infrastructure management.",
          C: "INCORRECT: Lambda has a 15-minute timeout and limited memory. Not suitable for large file transformations.",
          D: "INCORRECT: EC2 requires instance management, patching, and manual scheduling setup. High operational overhead."
        }
      },
      {
        id: 3,
        domain: "Data Engineering",
        question: "A machine learning team has a 200GB dataset in S3. Training jobs using SageMaker's built-in XGBoost algorithm are taking too long to start. What should be done to reduce startup time?",
        options: [
          { id: "A", text: "Convert the data to RecordIO format" },
          { id: "B", text: "Use SageMaker Pipe mode instead of File mode" },
          { id: "C", text: "Increase the instance size" },
          { id: "D", text: "Enable S3 Transfer Acceleration" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: RecordIO format improves some algorithms but doesn't address the startup time issue caused by downloading data.",
          B: "CORRECT: Pipe mode streams data directly from S3 to the training algorithm without downloading the entire dataset first. This significantly reduces startup time for large datasets.",
          C: "INCORRECT: Larger instances don't reduce the time to download data from S3.",
          D: "INCORRECT: Transfer Acceleration improves upload speeds to S3, not download speeds to training instances."
        }
      },
      {
        id: 4,
        domain: "Data Engineering",
        question: "An ML team needs to store and serve features for both real-time inference (requiring single-digit millisecond latency) and batch model training. Which AWS service provides this capability natively?",
        options: [
          { id: "A", text: "Amazon DynamoDB with DynamoDB Streams" },
          { id: "B", text: "Amazon ElastiCache for Redis" },
          { id: "C", text: "Amazon SageMaker Feature Store" },
          { id: "D", text: "Amazon Redshift with Spectrum" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: DynamoDB provides low latency but lacks ML-specific features like feature versioning and offline store integration.",
          B: "INCORRECT: ElastiCache is fast for caching but doesn't provide offline storage for batch training or feature management capabilities.",
          C: "CORRECT: SageMaker Feature Store has an Online Store for low-latency serving and an Offline Store (S3-backed) for batch training. It automatically keeps both in sync.",
          D: "INCORRECT: Redshift is a data warehouse optimized for analytics queries, not low-latency feature serving."
        }
      },
      {
        id: 5,
        domain: "Data Engineering",
        question: "A company wants to migrate their on-premises MySQL database to AWS and use the data for ML training without impacting production database performance. Which approach is recommended?",
        options: [
          { id: "A", text: "Use AWS Database Migration Service (DMS) with change data capture to replicate to S3" },
          { id: "B", text: "Create read replicas in Amazon RDS and query directly for training" },
          { id: "C", text: "Export daily database dumps to S3 using cron jobs" },
          { id: "D", text: "Use AWS Direct Connect for real-time queries to on-premises database" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: DMS with CDC continuously replicates data to S3 with minimal impact on the source database. S3 is ideal for ML training data storage.",
          B: "INCORRECT: Read replicas still consume database resources and aren't optimized for ML training workloads.",
          C: "INCORRECT: Daily dumps create data staleness and put load on the database during export.",
          D: "INCORRECT: Direct Connect doesn't solve the problem of impacting production during queries."
        }
      },
      {
        id: 6,
        domain: "Data Engineering",
        question: "A data scientist wants to query data stored in S3 using SQL without loading it into a database. The queries will be ad-hoc and infrequent. Which service is most cost-effective?",
        options: [
          { id: "A", text: "Amazon Redshift" },
          { id: "B", text: "Amazon Athena" },
          { id: "C", text: "Amazon EMR with Hive" },
          { id: "D", text: "Amazon RDS" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Redshift requires provisioning a cluster with ongoing costs even when not querying.",
          B: "CORRECT: Athena is serverless and charges only for data scanned. Perfect for ad-hoc, infrequent queries on S3 data with no infrastructure to manage.",
          C: "INCORRECT: EMR requires cluster management and has costs even when idle.",
          D: "INCORRECT: RDS requires loading data into the database first, doesn't query S3 directly."
        }
      },
      {
        id: 7,
        domain: "Data Engineering",
        question: "A company processes streaming IoT sensor data and needs to detect anomalies in real-time before storing the data. Which architecture supports this requirement?",
        options: [
          { id: "A", text: "Kinesis Data Streams → Kinesis Data Analytics → Kinesis Data Firehose → S3" },
          { id: "B", text: "IoT Core → S3 → Lambda → SNS" },
          { id: "C", text: "API Gateway → SQS → Lambda → DynamoDB" },
          { id: "D", text: "Direct Connect → EC2 → RDS" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: Kinesis Data Analytics can run SQL or Apache Flink applications on streaming data for real-time anomaly detection. Results can be sent to Firehose for storage or alerts.",
          B: "INCORRECT: This architecture processes data after storage, not in real-time before storage.",
          C: "INCORRECT: SQS doesn't support real-time stream processing or anomaly detection.",
          D: "INCORRECT: This doesn't provide any streaming or anomaly detection capabilities."
        }
      },
      {
        id: 8,
        domain: "Data Engineering",
        question: "A Glue ETL job processes new files arriving in S3 daily. The job should only process new files and not reprocess files from previous runs. Which Glue feature enables this?",
        options: [
          { id: "A", text: "Glue Data Catalog" },
          { id: "B", text: "Glue Job Bookmarks" },
          { id: "C", text: "Glue Crawlers" },
          { id: "D", text: "Glue Workflows" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: The Data Catalog stores metadata about data sources, not processing state.",
          B: "CORRECT: Job Bookmarks track which data has been processed. When enabled, Glue automatically processes only new data in subsequent runs.",
          C: "INCORRECT: Crawlers discover schema and update the Data Catalog. They don't track processing state.",
          D: "INCORRECT: Workflows orchestrate multiple jobs but don't track individual file processing."
        }
      },
      {
        id: 9,
        domain: "Data Engineering",
        question: "An organization has data scientists using notebooks who need to prepare data for ML. They want a visual interface to explore, transform, and export data to various destinations. Which SageMaker feature should they use?",
        options: [
          { id: "A", text: "SageMaker Processing" },
          { id: "B", text: "SageMaker Data Wrangler" },
          { id: "C", text: "SageMaker Clarify" },
          { id: "D", text: "SageMaker Autopilot" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Processing runs data processing scripts but doesn't provide a visual interface.",
          B: "CORRECT: Data Wrangler provides a visual interface in SageMaker Studio with 300+ built-in transformations. Data can be exported to Processing jobs, Pipelines, or Feature Store.",
          C: "INCORRECT: Clarify is for bias detection and explainability, not data preparation.",
          D: "INCORRECT: Autopilot automatically builds ML models but isn't focused on data preparation."
        }
      },
      {
        id: 10,
        domain: "Data Engineering",
        question: "A large retailer needs to process 500TB of historical transaction data for ML model training. The processing requires custom Python libraries with C extensions. Which service is most appropriate?",
        options: [
          { id: "A", text: "AWS Lambda" },
          { id: "B", text: "AWS Glue" },
          { id: "C", text: "Amazon EMR with bootstrap actions" },
          { id: "D", text: "Amazon Athena" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Lambda has memory limits and 15-minute timeout, unsuitable for 500TB processing.",
          B: "INCORRECT: Glue has limitations with native C extension libraries and may struggle with custom dependencies.",
          C: "CORRECT: EMR handles massive scale (500TB+) and bootstrap actions allow installing any custom libraries including those with C extensions. Full control over the processing environment.",
          D: "INCORRECT: Athena is for SQL queries, not complex data transformations with custom libraries."
        }
      },
      {
        id: 11,
        domain: "Data Engineering",
        question: "Which S3 storage class should be used for ML training data that is accessed frequently during model development but rarely after models are deployed to production?",
        options: [
          { id: "A", text: "S3 Glacier Deep Archive" },
          { id: "B", text: "S3 Standard" },
          { id: "C", text: "S3 Intelligent-Tiering" },
          { id: "D", text: "S3 One Zone-IA" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Glacier Deep Archive has retrieval times of hours, unsuitable for ML training access patterns.",
          B: "INCORRECT: Standard is more expensive than necessary for data that becomes infrequently accessed.",
          C: "CORRECT: Intelligent-Tiering automatically moves data between access tiers based on usage patterns. Frequent access during development, then automatic cost optimization when access decreases.",
          D: "INCORRECT: One Zone-IA has lower availability and doesn't automatically tier based on access patterns."
        }
      },
      {
        id: 12,
        domain: "Data Engineering",
        question: "A data pipeline uses AWS Glue to process data from multiple sources. The team needs a central place to search, discover, and manage metadata about all datasets. Which component provides this?",
        options: [
          { id: "A", text: "AWS Glue Data Catalog" },
          { id: "B", text: "AWS Lake Formation" },
          { id: "C", text: "Amazon Macie" },
          { id: "D", text: "AWS Config" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: The Glue Data Catalog is a centralized metadata repository. It stores table definitions, schema, and location information that can be used by Glue, Athena, EMR, and Redshift Spectrum.",
          B: "INCORRECT: Lake Formation provides governance and security on top of the Data Catalog but isn't the metadata store itself.",
          C: "INCORRECT: Macie is for sensitive data discovery and security, not general metadata management.",
          D: "INCORRECT: Config tracks AWS resource configurations, not data catalog metadata."
        }
      },
      {
        id: 13,
        domain: "Data Engineering",
        question: "A SageMaker training job needs random access to specific records within a large dataset stored in S3, similar to reading files from a local file system. Which input mode should be used?",
        options: [
          { id: "A", text: "File mode" },
          { id: "B", text: "Pipe mode" },
          { id: "C", text: "FastFile mode" },
          { id: "D", text: "Streaming mode" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: File mode downloads the entire dataset before training starts, adding startup latency.",
          B: "INCORRECT: Pipe mode streams data sequentially, not optimized for random access patterns.",
          C: "CORRECT: FastFile mode exposes S3 data through a POSIX-compliant file system interface, allowing random access patterns while streaming data on-demand.",
          D: "INCORRECT: There is no 'Streaming mode' in SageMaker input modes."
        }
      },
      // Domain 2: Exploratory Data Analysis (16 questions - 24%)
      {
        id: 14,
        domain: "Exploratory Data Analysis",
        question: "A dataset has a numerical feature with values ranging from 1 to 1,000,000 and is heavily right-skewed. Which transformation would best normalize this distribution for a linear model?",
        options: [
          { id: "A", text: "Min-max scaling" },
          { id: "B", text: "Z-score standardization" },
          { id: "C", text: "Log transformation" },
          { id: "D", text: "Polynomial transformation" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Min-max scaling preserves the skewed distribution, just rescales to [0,1].",
          B: "INCORRECT: Z-score standardization doesn't address skewness, only centers and scales.",
          C: "CORRECT: Log transformation compresses large values and expands small values, reducing right skewness and making the distribution more normal.",
          D: "INCORRECT: Polynomial transformation would amplify skewness, not reduce it."
        }
      },
      {
        id: 15,
        domain: "Exploratory Data Analysis",
        question: "A classification model needs to predict customer churn. The dataset has 5% churners and 95% non-churners. What approach should be used to handle this imbalance?",
        options: [
          { id: "A", text: "Increase the learning rate" },
          { id: "B", text: "Add more features" },
          { id: "C", text: "Use SMOTE or class weights" },
          { id: "D", text: "Reduce model complexity" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Learning rate affects training dynamics, not class imbalance.",
          B: "INCORRECT: More features don't address the fundamental class imbalance problem.",
          C: "CORRECT: SMOTE generates synthetic minority class samples. Class weights penalize misclassification of the minority class more heavily. Both help the model learn from imbalanced data.",
          D: "INCORRECT: Model complexity is unrelated to class imbalance."
        }
      },
      {
        id: 16,
        domain: "Exploratory Data Analysis",
        question: "A text classification task requires converting documents to numerical features. The goal is to give higher weight to words that are distinctive to specific documents. Which method should be used?",
        options: [
          { id: "A", text: "Bag of Words" },
          { id: "B", text: "TF-IDF" },
          { id: "C", text: "One-hot encoding" },
          { id: "D", text: "Label encoding" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Bag of Words uses raw word counts without weighting by distinctiveness.",
          B: "CORRECT: TF-IDF (Term Frequency-Inverse Document Frequency) multiplies term frequency by inverse document frequency, upweighting rare/distinctive words and downweighting common words.",
          C: "INCORRECT: One-hot encoding creates binary indicators without any frequency or importance weighting.",
          D: "INCORRECT: Label encoding is for categorical targets, not text features."
        }
      },
      {
        id: 17,
        domain: "Exploratory Data Analysis",
        question: "A feature 'day_of_week' (1-7) is being used in a neural network. Using raw integer values would imply that day 7 is far from day 1. How should this cyclical feature be encoded?",
        options: [
          { id: "A", text: "One-hot encode into 7 features" },
          { id: "B", text: "Normalize to [0, 1]" },
          { id: "C", text: "Use sine and cosine transformations" },
          { id: "D", text: "Use label encoding" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: One-hot encoding doesn't capture the cyclical relationship where day 7 is adjacent to day 1.",
          B: "INCORRECT: Normalizing still implies day 7 (≈1) is far from day 1 (≈0).",
          C: "CORRECT: sin(2π×day/7) and cos(2π×day/7) create a circular representation where adjacent days have similar values, including day 7 and day 1.",
          D: "INCORRECT: Label encoding implies ordinal relationship without cyclical property."
        }
      },
      {
        id: 18,
        domain: "Exploratory Data Analysis",
        question: "A dataset contains 10% missing values in a critical feature. The missing values appear to be Missing Completely at Random (MCAR). Which imputation method is most appropriate for a dataset used with a tree-based model?",
        options: [
          { id: "A", text: "Delete all rows with missing values" },
          { id: "B", text: "Replace with the global mean" },
          { id: "C", text: "Replace with the median" },
          { id: "D", text: "Use a flag feature to indicate missingness and impute with median" }
        ],
        correct: "D",
        explanations: {
          A: "INCORRECT: Deleting 10% of data loses significant information.",
          B: "INCORRECT: Mean imputation can be affected by outliers and doesn't flag missingness.",
          C: "INCORRECT: Median is more robust but alone doesn't capture that the value was missing.",
          D: "CORRECT: Adding a binary flag indicating missingness lets tree-based models learn if missingness itself is predictive. Median imputation is robust to outliers for the actual value."
        }
      },
      {
        id: 19,
        domain: "Exploratory Data Analysis",
        question: "Two features in a linear regression model have a Pearson correlation coefficient of 0.92. What problem does this indicate and what is a solution?",
        options: [
          { id: "A", text: "Overfitting; add regularization" },
          { id: "B", text: "Multicollinearity; remove one feature or use PCA" },
          { id: "C", text: "Heteroscedasticity; transform the target" },
          { id: "D", text: "Autocorrelation; use time series differencing" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: High correlation indicates multicollinearity, not necessarily overfitting.",
          B: "CORRECT: Correlation of 0.92 indicates multicollinearity—features provide redundant information. This causes unstable coefficients in linear models. Remove one feature or use PCA to combine them.",
          C: "INCORRECT: Heteroscedasticity is about variance in residuals, not feature correlation.",
          D: "INCORRECT: Autocorrelation is about correlation in time series residuals, not feature correlation."
        }
      },
      {
        id: 20,
        domain: "Exploratory Data Analysis",
        question: "A categorical feature 'product_category' has 500 unique values. The ML model is gradient boosting (XGBoost). Which encoding method is most suitable?",
        options: [
          { id: "A", text: "One-hot encoding" },
          { id: "B", text: "Label encoding" },
          { id: "C", text: "Target encoding" },
          { id: "D", text: "Binary encoding" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: One-hot encoding creates 500 sparse features, making training slow and memory-intensive.",
          B: "INCORRECT: Label encoding assigns arbitrary integers implying false ordinal relationships.",
          C: "CORRECT: Target encoding replaces each category with the mean target value, reducing 500 categories to 1 feature while capturing predictive information. Works well with tree models.",
          D: "INCORRECT: Binary encoding creates ~9 features (log2(500)) but loses semantic meaning."
        }
      },
      {
        id: 21,
        domain: "Exploratory Data Analysis",
        question: "A dataset shows that when feature X increases, feature Y decreases proportionally. The Pearson correlation is -0.95. What does this indicate?",
        options: [
          { id: "A", text: "X and Y have a strong positive linear relationship" },
          { id: "B", text: "X and Y have a strong negative linear relationship" },
          { id: "C", text: "X and Y have no relationship" },
          { id: "D", text: "X and Y have a non-linear relationship" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: -0.95 is negative, indicating inverse relationship.",
          B: "CORRECT: A correlation of -0.95 indicates a strong negative linear relationship—as X increases, Y decreases proportionally.",
          C: "INCORRECT: Correlation near 0 would indicate no linear relationship.",
          D: "INCORRECT: High absolute correlation indicates a strong linear relationship, not non-linear."
        }
      },
      {
        id: 22,
        domain: "Exploratory Data Analysis",
        question: "A fraud detection model will be used where catching fraud is critical, but false positives are acceptable. Which metric should be prioritized?",
        options: [
          { id: "A", text: "Precision" },
          { id: "B", text: "Specificity" },
          { id: "C", text: "Recall" },
          { id: "D", text: "Accuracy" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Precision prioritizes reducing false positives, which is acceptable in this case.",
          B: "INCORRECT: Specificity measures true negative rate, less relevant when catching positives is priority.",
          C: "CORRECT: Recall = TP/(TP+FN) measures what percentage of actual fraud cases are caught. High recall ensures most fraud is detected, even with some false positives.",
          D: "INCORRECT: Accuracy doesn't distinguish between error types and is misleading with imbalanced classes."
        }
      },
      {
        id: 23,
        domain: "Exploratory Data Analysis",
        question: "Before applying PCA for dimensionality reduction, what preprocessing step is essential?",
        options: [
          { id: "A", text: "One-hot encoding" },
          { id: "B", text: "Standardization (z-score normalization)" },
          { id: "C", text: "Log transformation" },
          { id: "D", text: "Target encoding" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: One-hot encoding is for categorical variables; PCA works on numerical data.",
          B: "CORRECT: PCA is sensitive to feature scale. Features with larger variance dominate the principal components. Standardization ensures all features contribute equally.",
          C: "INCORRECT: Log transformation addresses skewness but isn't specifically required for PCA.",
          D: "INCORRECT: Target encoding is for categorical features, not PCA preprocessing."
        }
      },
      {
        id: 24,
        domain: "Exploratory Data Analysis",
        question: "A model achieves 99% accuracy on a dataset with 99% negative and 1% positive examples. What does this suggest?",
        options: [
          { id: "A", text: "The model is excellent" },
          { id: "B", text: "The model may be predicting all negative and not learning" },
          { id: "C", text: "More training data is needed" },
          { id: "D", text: "The model is overfitting" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: 99% accuracy equals the majority class proportion—not indicative of learning.",
          B: "CORRECT: A model predicting all negative would achieve 99% accuracy on this dataset. The accuracy metric is misleading with severe class imbalance. Check precision, recall, and confusion matrix.",
          C: "INCORRECT: More data won't help if the model isn't learning the minority class.",
          D: "INCORRECT: This pattern indicates the model may not be learning, not necessarily overfitting."
        }
      },
      {
        id: 25,
        domain: "Exploratory Data Analysis",
        question: "Feature selection needs to be performed on a dataset with 1000 features. The goal is to identify features with statistical dependency on the target without training a model. Which method should be used?",
        options: [
          { id: "A", text: "Recursive Feature Elimination (RFE)" },
          { id: "B", text: "L1 Regularization" },
          { id: "C", text: "Mutual Information" },
          { id: "D", text: "Permutation Importance" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: RFE requires training models iteratively—a wrapper method, not filter method.",
          B: "INCORRECT: L1 regularization is an embedded method requiring model training.",
          C: "CORRECT: Mutual Information measures statistical dependency between features and target without training a model. It's a filter method that works on the data directly.",
          D: "INCORRECT: Permutation Importance requires a trained model to measure importance."
        }
      },
      {
        id: 26,
        domain: "Exploratory Data Analysis",
        question: "A dataset has extreme outliers in a feature (values 100x the median). Which method preserves most data while reducing outlier impact?",
        options: [
          { id: "A", text: "Remove all outliers" },
          { id: "B", text: "Winsorization (capping at percentiles)" },
          { id: "C", text: "Replace with mean" },
          { id: "D", text: "Ignore the feature" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Removing outliers loses data and may remove legitimate extreme cases.",
          B: "CORRECT: Winsorization caps values at specified percentiles (e.g., 5th and 95th). This preserves all records while limiting outlier influence.",
          C: "INCORRECT: Replacing with mean would change legitimate values and is affected by the outliers themselves.",
          D: "INCORRECT: Ignoring the feature loses potentially valuable information."
        }
      },
      {
        id: 27,
        domain: "Exploratory Data Analysis",
        question: "When preparing training and test sets from time series data, how should the split be performed?",
        options: [
          { id: "A", text: "Random stratified sampling" },
          { id: "B", text: "K-fold cross-validation" },
          { id: "C", text: "Temporal split (train on past, test on future)" },
          { id: "D", text: "Bootstrap sampling" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Random sampling causes data leakage—model sees future data during training.",
          B: "INCORRECT: Standard k-fold mixes time periods, causing temporal leakage.",
          C: "CORRECT: Temporal split ensures the model trains only on past data and tests on future data, simulating real-world deployment where you predict the future.",
          D: "INCORRECT: Bootstrap sampling doesn't maintain temporal ordering."
        }
      },
      {
        id: 28,
        domain: "Exploratory Data Analysis",
        question: "A feature engineered from the target variable is accidentally included in the training set. What problem does this cause?",
        options: [
          { id: "A", text: "Underfitting" },
          { id: "B", text: "Data leakage" },
          { id: "C", text: "Class imbalance" },
          { id: "D", text: "Multicollinearity" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Including target information would cause overly good performance, not underfitting.",
          B: "CORRECT: Data leakage occurs when information from the target variable is included in features. The model appears to perform well but fails in production where this information isn't available.",
          C: "INCORRECT: Class imbalance is about target distribution, not feature engineering errors.",
          D: "INCORRECT: Multicollinearity is correlation between features, not target leakage."
        }
      },
      {
        id: 29,
        domain: "Exploratory Data Analysis",
        question: "For visualizing high-dimensional data (50 features) in 2D for exploration, which technique is most appropriate?",
        options: [
          { id: "A", text: "PCA (Principal Component Analysis)" },
          { id: "B", text: "t-SNE (t-Distributed Stochastic Neighbor Embedding)" },
          { id: "C", text: "Linear Discriminant Analysis" },
          { id: "D", text: "Random projection" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: PCA preserves global variance but may not reveal cluster structure in 2D effectively.",
          B: "CORRECT: t-SNE is specifically designed for 2D/3D visualization of high-dimensional data. It preserves local neighborhood structure, making clusters visible.",
          C: "INCORRECT: LDA is supervised and focuses on class separation, not general exploration.",
          D: "INCORRECT: Random projection is for dimensionality reduction but not optimized for visualization."
        }
      },
      // Domain 3: Modeling (23 questions - 36%)
      {
        id: 30,
        domain: "Modeling",
        question: "A SageMaker XGBoost model shows low training error but high validation error that increases over epochs. What is happening and what should be done?",
        options: [
          { id: "A", text: "Underfitting; increase max_depth" },
          { id: "B", text: "Overfitting; decrease max_depth and increase min_child_weight" },
          { id: "C", text: "Data leakage; check feature engineering" },
          { id: "D", text: "Class imbalance; use SMOTE" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Low training error with high validation error is overfitting, not underfitting.",
          B: "CORRECT: This is classic overfitting. Reducing max_depth creates simpler trees. Increasing min_child_weight requires more samples per leaf, both reducing overfitting.",
          C: "INCORRECT: Data leakage would show unrealistically good validation performance, not poor.",
          D: "INCORRECT: Class imbalance doesn't cause this specific pattern of diverging train/validation error."
        }
      },
      {
        id: 31,
        domain: "Modeling",
        question: "A multi-class classification problem has 10 mutually exclusive classes. What activation function and loss should be used in the output layer?",
        options: [
          { id: "A", text: "Sigmoid with binary cross-entropy" },
          { id: "B", text: "Softmax with categorical cross-entropy" },
          { id: "C", text: "ReLU with MSE" },
          { id: "D", text: "Tanh with hinge loss" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Sigmoid with binary cross-entropy is for binary or multi-label (non-mutually exclusive) classification.",
          B: "CORRECT: Softmax produces probabilities summing to 1, appropriate for mutually exclusive classes. Categorical cross-entropy is the standard loss for multi-class classification.",
          C: "INCORRECT: ReLU is for hidden layers, MSE is for regression.",
          D: "INCORRECT: Tanh outputs [-1,1], not probabilities. Hinge loss is used with SVMs."
        }
      },
      {
        id: 32,
        domain: "Modeling",
        question: "A problem requires predicting multiple non-mutually exclusive labels per image (e.g., 'has_cat', 'has_dog', 'is_outdoor'). What output configuration should be used?",
        options: [
          { id: "A", text: "Single softmax output" },
          { id: "B", text: "Multiple sigmoid outputs with binary cross-entropy" },
          { id: "C", text: "Single linear output" },
          { id: "D", text: "Argmax layer" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Softmax assumes mutually exclusive classes—only one can be true.",
          B: "CORRECT: Multiple sigmoid outputs allow independent probability for each label. Binary cross-entropy loss for each. This handles multi-label classification where multiple labels can be true.",
          C: "INCORRECT: Linear output is for regression, not classification.",
          D: "INCORRECT: Argmax is used for inference to select the highest probability class, not as a layer configuration."
        }
      },
      {
        id: 33,
        domain: "Modeling",
        question: "An NLP model needs to be trained for sentiment analysis but only 1,000 labeled examples are available. Which approach is most likely to succeed?",
        options: [
          { id: "A", text: "Train a custom LSTM from scratch" },
          { id: "B", text: "Use a pre-trained BERT model and fine-tune" },
          { id: "C", text: "Use bag of words with naive Bayes" },
          { id: "D", text: "Train a transformer from scratch" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: 1,000 examples is insufficient to train an LSTM from scratch without overfitting.",
          B: "CORRECT: Transfer learning with pre-trained BERT leverages knowledge from massive text corpora. Fine-tuning on 1,000 examples often achieves strong results.",
          C: "INCORRECT: Bag of words with naive Bayes is simple but loses word order and semantic meaning.",
          D: "INCORRECT: Transformers have millions of parameters—far too many for 1,000 examples."
        }
      },
      {
        id: 34,
        domain: "Modeling",
        question: "A recommendation system needs to be built, but there's no historical user interaction data. Only item descriptions are available. Which approach should be used?",
        options: [
          { id: "A", text: "Collaborative filtering" },
          { id: "B", text: "Matrix factorization" },
          { id: "C", text: "Content-based filtering" },
          { id: "D", text: "User-based nearest neighbors" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Collaborative filtering requires user-item interaction history.",
          B: "INCORRECT: Matrix factorization needs a user-item interaction matrix.",
          C: "CORRECT: Content-based filtering uses item features (descriptions) to recommend similar items. Works without user interaction history—solves the cold-start problem.",
          D: "INCORRECT: User-based methods need historical user behavior to find similar users."
        }
      },
      {
        id: 35,
        domain: "Modeling",
        question: "A SageMaker training job needs to be configured for an algorithm that isn't available as a built-in. Which approach allows using any ML framework?",
        options: [
          { id: "A", text: "SageMaker Autopilot" },
          { id: "B", text: "SageMaker Bring Your Own Container (BYOC)" },
          { id: "C", text: "SageMaker JumpStart" },
          { id: "D", text: "SageMaker Experiments" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Autopilot automatically builds models using built-in algorithms, not custom frameworks.",
          B: "CORRECT: BYOC allows packaging any ML framework in a Docker container. Full control over the training environment.",
          C: "INCORRECT: JumpStart provides pre-built solutions and models, not custom framework support.",
          D: "INCORRECT: Experiments tracks training runs but doesn't provide custom framework support."
        }
      },
      {
        id: 36,
        domain: "Modeling",
        question: "For a binary classification problem, which SageMaker built-in algorithm would be most appropriate as a starting point?",
        options: [
          { id: "A", text: "DeepAR" },
          { id: "B", text: "K-Means" },
          { id: "C", text: "XGBoost or Linear Learner" },
          { id: "D", text: "Random Cut Forest" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: DeepAR is for time series forecasting.",
          B: "INCORRECT: K-Means is unsupervised clustering, not classification.",
          C: "CORRECT: Both XGBoost and Linear Learner support binary classification. XGBoost for complex patterns, Linear Learner for simpler linear relationships.",
          D: "INCORRECT: Random Cut Forest is for anomaly detection, not classification."
        }
      },
      {
        id: 37,
        domain: "Modeling",
        question: "Time series data for 500 products needs to be forecasted. The products have similar seasonal patterns. Which SageMaker algorithm is best suited?",
        options: [
          { id: "A", text: "Linear Learner" },
          { id: "B", text: "XGBoost" },
          { id: "C", text: "DeepAR" },
          { id: "D", text: "BlazingText" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Linear Learner isn't specialized for time series forecasting.",
          B: "INCORRECT: XGBoost can do time series with manual feature engineering but isn't designed for multiple related series.",
          C: "CORRECT: DeepAR is designed for forecasting multiple related time series. It learns patterns across all series simultaneously, handling seasonality effectively.",
          D: "INCORRECT: BlazingText is for text classification and Word2Vec, not time series."
        }
      },
      {
        id: 38,
        domain: "Modeling",
        question: "A streaming anomaly detection system needs to identify unusual patterns in real-time data. Which SageMaker algorithm is designed for this?",
        options: [
          { id: "A", text: "XGBoost" },
          { id: "B", text: "Random Cut Forest" },
          { id: "C", text: "K-Means" },
          { id: "D", text: "Object Detection" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: XGBoost is supervised and requires labeled anomaly data for training.",
          B: "CORRECT: Random Cut Forest is an unsupervised algorithm specifically designed for anomaly detection. It assigns anomaly scores based on how easily points can be isolated.",
          C: "INCORRECT: K-Means clusters data but doesn't directly detect anomalies.",
          D: "INCORRECT: Object Detection is for finding objects in images."
        }
      },
      {
        id: 39,
        domain: "Modeling",
        question: "During training, both training loss and validation loss are decreasing steadily. What does this indicate?",
        options: [
          { id: "A", text: "The model is overfitting" },
          { id: "B", text: "The model is underfitting" },
          { id: "C", text: "The model is learning well and should continue training" },
          { id: "D", text: "The learning rate is too high" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Overfitting shows validation loss increasing while training loss decreases.",
          B: "INCORRECT: Underfitting would show both losses remaining high.",
          C: "CORRECT: Both losses decreasing indicates healthy learning. Continue training until validation loss plateaus or starts increasing.",
          D: "INCORRECT: Too-high learning rate would show unstable or oscillating losses."
        }
      },
      {
        id: 40,
        domain: "Modeling",
        question: "What regularization technique randomly sets neurons to zero during training to prevent co-adaptation?",
        options: [
          { id: "A", text: "L1 regularization" },
          { id: "B", text: "L2 regularization" },
          { id: "C", text: "Dropout" },
          { id: "D", text: "Batch normalization" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: L1 regularization adds absolute value of weights to loss, encouraging sparsity.",
          B: "INCORRECT: L2 regularization adds squared weights to loss, preventing large weights.",
          C: "CORRECT: Dropout randomly sets neurons to zero during training, preventing neurons from co-adapting and improving generalization.",
          D: "INCORRECT: Batch normalization normalizes layer inputs but doesn't zero neurons."
        }
      },
      {
        id: 41,
        domain: "Modeling",
        question: "Which regularization technique produces sparse models by driving some feature weights exactly to zero?",
        options: [
          { id: "A", text: "L1 regularization (Lasso)" },
          { id: "B", text: "L2 regularization (Ridge)" },
          { id: "C", text: "Dropout" },
          { id: "D", text: "Early stopping" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: L1 regularization can drive weights exactly to zero, effectively performing feature selection and creating sparse models.",
          B: "INCORRECT: L2 regularization shrinks weights toward zero but rarely makes them exactly zero.",
          C: "INCORRECT: Dropout randomly zeros activations during training but doesn't affect final weights.",
          D: "INCORRECT: Early stopping halts training but doesn't create sparse weights."
        }
      },
      {
        id: 42,
        domain: "Modeling",
        question: "A neural network for image classification isn't converging—loss oscillates wildly. What should be tried first?",
        options: [
          { id: "A", text: "Increase learning rate" },
          { id: "B", text: "Decrease learning rate" },
          { id: "C", text: "Add more layers" },
          { id: "D", text: "Remove dropout" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Higher learning rate would cause more oscillation.",
          B: "CORRECT: Oscillating loss is a classic sign of learning rate too high. Decreasing it allows more stable gradient descent.",
          C: "INCORRECT: Model architecture isn't the issue when loss oscillates.",
          D: "INCORRECT: Dropout helps regularization but doesn't cause loss oscillation."
        }
      },
      {
        id: 43,
        domain: "Modeling",
        question: "Which optimizer is most commonly used as a default choice for deep learning due to its adaptive learning rates?",
        options: [
          { id: "A", text: "SGD (Stochastic Gradient Descent)" },
          { id: "B", text: "Adam" },
          { id: "C", text: "Adagrad" },
          { id: "D", text: "Momentum" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: SGD is effective but requires careful learning rate tuning.",
          B: "CORRECT: Adam combines momentum and adaptive learning rates. It's robust, works well with default parameters, and is the most popular choice.",
          C: "INCORRECT: Adagrad's learning rate can decrease too much over time.",
          D: "INCORRECT: Momentum improves SGD but isn't as adaptive as Adam."
        }
      },
      {
        id: 44,
        domain: "Modeling",
        question: "A medical imaging model needs to detect tumors in X-rays. Limited labeled data is available. Which approach leverages knowledge from other image recognition tasks?",
        options: [
          { id: "A", text: "Train a CNN from scratch" },
          { id: "B", text: "Use transfer learning with pre-trained ImageNet weights" },
          { id: "C", text: "Use random initialization" },
          { id: "D", text: "Use unsupervised clustering" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Training from scratch with limited data leads to overfitting.",
          B: "CORRECT: Transfer learning uses a model pre-trained on ImageNet, which has learned general image features. Fine-tuning on medical images requires much less data.",
          C: "INCORRECT: Random initialization provides no prior knowledge advantage.",
          D: "INCORRECT: Unsupervised clustering doesn't leverage labeled image knowledge for detection."
        }
      },
      {
        id: 45,
        domain: "Modeling",
        question: "For a regression problem where the target is always positive (e.g., prices), what transformation might improve a linear model?",
        options: [
          { id: "A", text: "Predict log(target) instead of target" },
          { id: "B", text: "Use one-hot encoding on the target" },
          { id: "C", text: "Discretize the target into bins" },
          { id: "D", text: "Use softmax output" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: Predicting log(target) ensures predictions are positive after exp() transformation. Also helps with right-skewed targets common in prices.",
          B: "INCORRECT: One-hot encoding converts to classification, losing the continuous nature.",
          C: "INCORRECT: Discretizing loses precision and converts to ordinal classification.",
          D: "INCORRECT: Softmax is for multi-class classification, not regression."
        }
      },
      {
        id: 46,
        domain: "Modeling",
        question: "What does the IoU (Intersection over Union) metric measure in object detection?",
        options: [
          { id: "A", text: "Classification accuracy" },
          { id: "B", text: "Overlap between predicted and ground truth bounding boxes" },
          { id: "C", text: "Number of objects detected" },
          { id: "D", text: "Processing speed" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: IoU doesn't measure classification accuracy.",
          B: "CORRECT: IoU = (Intersection Area) / (Union Area) of predicted and ground truth boxes. It measures how well the predicted bounding box aligns with the actual object location.",
          C: "INCORRECT: IoU measures box overlap quality, not count.",
          D: "INCORRECT: IoU is an accuracy metric, not a speed metric."
        }
      },
      {
        id: 47,
        domain: "Modeling",
        question: "SageMaker Hyperparameter Tuning uses which strategy by default to efficiently search the hyperparameter space?",
        options: [
          { id: "A", text: "Grid search" },
          { id: "B", text: "Random search" },
          { id: "C", text: "Bayesian optimization" },
          { id: "D", text: "Manual tuning" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Grid search evaluates all combinations, inefficient for large spaces.",
          B: "INCORRECT: Random search is available but not the default.",
          C: "CORRECT: SageMaker uses Bayesian optimization by default. It builds a probabilistic model of the objective function and intelligently selects next hyperparameters based on previous results.",
          D: "INCORRECT: Manual tuning isn't an automated strategy."
        }
      },
      {
        id: 48,
        domain: "Modeling",
        question: "A data scientist wants to quickly get a baseline ML model for a tabular dataset without writing code. Which SageMaker feature automatically builds and trains models?",
        options: [
          { id: "A", text: "SageMaker Debugger" },
          { id: "B", text: "SageMaker Autopilot" },
          { id: "C", text: "SageMaker Clarify" },
          { id: "D", text: "SageMaker Model Monitor" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Debugger analyzes training runs, doesn't automatically build models.",
          B: "CORRECT: Autopilot automatically explores data, selects algorithms, trains models, and provides the best model with notebook code. Ideal for quick baseline models.",
          C: "INCORRECT: Clarify provides bias detection and explainability, not automatic model building.",
          D: "INCORRECT: Model Monitor tracks deployed models, doesn't build them."
        }
      },
      {
        id: 49,
        domain: "Modeling",
        question: "A classification model needs to output well-calibrated probabilities (e.g., 80% confidence should be correct 80% of the time). What technique helps achieve this?",
        options: [
          { id: "A", text: "Probability calibration (Platt scaling or isotonic regression)" },
          { id: "B", text: "Feature scaling" },
          { id: "C", text: "Hyperparameter tuning" },
          { id: "D", text: "Cross-validation" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: Probability calibration adjusts model outputs so predicted probabilities match actual frequencies. Platt scaling fits a sigmoid, isotonic regression fits a non-decreasing function.",
          B: "INCORRECT: Feature scaling affects input preprocessing, not output probabilities.",
          C: "INCORRECT: Hyperparameter tuning optimizes performance metrics, not probability calibration.",
          D: "INCORRECT: Cross-validation evaluates models but doesn't calibrate probabilities."
        }
      },
      {
        id: 50,
        domain: "Modeling",
        question: "A model for predicting customer lifetime value needs to handle the fact that some customers haven't fully matured yet (censored data). Which approach is appropriate?",
        options: [
          { id: "A", text: "Standard linear regression" },
          { id: "B", text: "Survival analysis" },
          { id: "C", text: "Classification into bins" },
          { id: "D", text: "Remove censored observations" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Standard regression doesn't handle censoring—would underestimate value for immature customers.",
          B: "CORRECT: Survival analysis explicitly models time-to-event data with censoring. It properly handles customers who haven't yet churned or fully developed.",
          C: "INCORRECT: Classification loses continuous value information and doesn't handle censoring.",
          D: "INCORRECT: Removing censored data loses valuable information and biases results."
        }
      },
      {
        id: 51,
        domain: "Modeling",
        question: "Which loss function is most appropriate for a regression task?",
        options: [
          { id: "A", text: "Cross-entropy loss" },
          { id: "B", text: "Hinge loss" },
          { id: "C", text: "Mean Squared Error (MSE)" },
          { id: "D", text: "Softmax loss" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Cross-entropy is for classification tasks.",
          B: "INCORRECT: Hinge loss is used with SVMs for classification.",
          C: "CORRECT: MSE measures average squared difference between predictions and targets—standard for regression.",
          D: "INCORRECT: Softmax loss (cross-entropy after softmax) is for multi-class classification."
        }
      },
      {
        id: 52,
        domain: "Modeling",
        question: "In gradient boosting, what does 'boosting' refer to?",
        options: [
          { id: "A", text: "Training multiple models independently and averaging" },
          { id: "B", text: "Training models sequentially, each correcting errors of previous models" },
          { id: "C", text: "Using multiple GPUs for parallel training" },
          { id: "D", text: "Increasing batch size during training" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Training independently and averaging is bagging, not boosting.",
          B: "CORRECT: Boosting trains models sequentially. Each new model focuses on examples the previous models got wrong, gradually improving overall performance.",
          C: "INCORRECT: Parallel hardware usage isn't related to boosting.",
          D: "INCORRECT: Batch size is unrelated to boosting."
        }
      },
      // Domain 4: ML Implementation and Operations (13 questions - 20%)
      {
        id: 53,
        domain: "ML Implementation and Operations",
        question: "A model deployment needs to handle variable traffic with zero cost during idle periods. Which SageMaker deployment option is best?",
        options: [
          { id: "A", text: "Real-time endpoint with auto-scaling" },
          { id: "B", text: "Serverless inference" },
          { id: "C", text: "Batch transform" },
          { id: "D", text: "Asynchronous inference" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Real-time endpoints always have minimum instances running, incurring costs during idle time.",
          B: "CORRECT: Serverless inference scales to zero when there's no traffic—zero cost during idle periods. Automatically scales up when requests arrive.",
          C: "INCORRECT: Batch transform is for processing large batches, not on-demand requests.",
          D: "INCORRECT: Asynchronous inference is for large payloads with long processing, not cost optimization."
        }
      },
      {
        id: 54,
        domain: "ML Implementation and Operations",
        question: "A deployed model's accuracy has degraded over time. Monitoring shows the input data distribution has shifted from the training data. What is this called?",
        options: [
          { id: "A", text: "Overfitting" },
          { id: "B", text: "Data drift" },
          { id: "C", text: "Model bias" },
          { id: "D", text: "Feature leakage" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Overfitting is poor generalization at training time, not degradation in production.",
          B: "CORRECT: Data drift occurs when production data distribution differs from training data. The model was trained on old patterns that no longer match current data.",
          C: "INCORRECT: Model bias is systematic unfairness, not distribution shift.",
          D: "INCORRECT: Feature leakage is including target information in features during training."
        }
      },
      {
        id: 55,
        domain: "ML Implementation and Operations",
        question: "Which SageMaker feature continuously monitors production models for data quality issues, model quality degradation, and bias drift?",
        options: [
          { id: "A", text: "SageMaker Debugger" },
          { id: "B", text: "SageMaker Model Monitor" },
          { id: "C", text: "SageMaker Experiments" },
          { id: "D", text: "SageMaker Pipelines" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Debugger analyzes training jobs, not production monitoring.",
          B: "CORRECT: Model Monitor continuously evaluates production traffic against baselines, detecting data quality issues, model accuracy degradation, and bias drift.",
          C: "INCORRECT: Experiments tracks training runs, not production monitoring.",
          D: "INCORRECT: Pipelines orchestrate ML workflows, not monitor deployed models."
        }
      },
      {
        id: 56,
        domain: "ML Implementation and Operations",
        question: "A company needs to deploy a new model version with minimal risk. They want to send 10% of traffic to the new version initially. Which strategy should they use?",
        options: [
          { id: "A", text: "Blue/Green deployment with immediate cutover" },
          { id: "B", text: "Canary deployment" },
          { id: "C", text: "Rolling deployment" },
          { id: "D", text: "Recreate deployment" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Blue/Green with immediate cutover sends 100% traffic to new version—higher risk.",
          B: "CORRECT: Canary deployment routes a small percentage (10%) to the new version. If metrics look good, gradually increase traffic. Minimizes blast radius.",
          C: "INCORRECT: Rolling updates instances gradually but doesn't offer precise traffic control.",
          D: "INCORRECT: Recreate deployment causes downtime and has no gradual rollout."
        }
      },
      {
        id: 57,
        domain: "ML Implementation and Operations",
        question: "A financial services company needs to explain individual predictions to comply with regulations. Which SageMaker capability should be used?",
        options: [
          { id: "A", text: "SageMaker Debugger" },
          { id: "B", text: "SageMaker Model Monitor" },
          { id: "C", text: "SageMaker Clarify" },
          { id: "D", text: "SageMaker Neo" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: Debugger analyzes training, not individual predictions.",
          B: "INCORRECT: Model Monitor tracks aggregate metrics, not individual prediction explanations.",
          C: "CORRECT: SageMaker Clarify provides SHAP-based feature attributions explaining which factors contributed to each prediction—essential for regulatory compliance.",
          D: "INCORRECT: Neo optimizes models for deployment, doesn't explain predictions."
        }
      },
      {
        id: 58,
        domain: "ML Implementation and Operations",
        question: "A SageMaker training job needs to be as cost-effective as possible. The job can tolerate interruptions. Which instance purchasing option should be used?",
        options: [
          { id: "A", text: "On-demand instances" },
          { id: "B", text: "Reserved instances" },
          { id: "C", text: "Spot instances" },
          { id: "D", text: "Dedicated hosts" }
        ],
        correct: "C",
        explanations: {
          A: "INCORRECT: On-demand has the highest cost.",
          B: "INCORRECT: Reserved requires upfront commitment and isn't as cost-effective for interruptible workloads.",
          C: "CORRECT: Spot instances offer up to 90% savings. For interruptible training jobs, enable checkpointing to resume if interrupted.",
          D: "INCORRECT: Dedicated hosts are the most expensive option."
        }
      },
      {
        id: 59,
        domain: "ML Implementation and Operations",
        question: "A model needs to be optimized for deployment on edge devices with limited compute. Which SageMaker service compiles models for target hardware?",
        options: [
          { id: "A", text: "SageMaker Inference Recommender" },
          { id: "B", text: "SageMaker Neo" },
          { id: "C", text: "SageMaker Model Registry" },
          { id: "D", text: "SageMaker Feature Store" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Inference Recommender helps select instance types, not compile for edge devices.",
          B: "CORRECT: SageMaker Neo compiles models for specific target hardware, optimizing for edge devices, specific CPUs/GPUs, reducing size and improving inference speed.",
          C: "INCORRECT: Model Registry stores and versions models, doesn't compile them.",
          D: "INCORRECT: Feature Store manages features, not model compilation."
        }
      },
      {
        id: 60,
        domain: "ML Implementation and Operations",
        question: "Multiple ML models need to be hosted on a single endpoint to reduce costs. Which SageMaker feature enables this?",
        options: [
          { id: "A", text: "Multi-model endpoint" },
          { id: "B", text: "Model parallelism" },
          { id: "C", text: "Data parallelism" },
          { id: "D", text: "Inference pipeline" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: Multi-model endpoints host multiple models on shared infrastructure, dynamically loading models as needed. Reduces cost compared to separate endpoints per model.",
          B: "INCORRECT: Model parallelism splits one large model across GPUs, not multiple models.",
          C: "INCORRECT: Data parallelism distributes training data, not inference models.",
          D: "INCORRECT: Inference pipelines chain preprocessing/postprocessing with a single model."
        }
      },
      {
        id: 61,
        domain: "ML Implementation and Operations",
        question: "An ML workflow needs to be automated end-to-end: data processing, training, evaluation, and conditional deployment. Which SageMaker service orchestrates this?",
        options: [
          { id: "A", text: "SageMaker Experiments" },
          { id: "B", text: "SageMaker Pipelines" },
          { id: "C", text: "SageMaker Debugger" },
          { id: "D", text: "SageMaker Autopilot" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Experiments tracks and compares runs but doesn't orchestrate workflows.",
          B: "CORRECT: SageMaker Pipelines defines and automates ML workflows with steps for processing, training, evaluation, and conditional logic for deployment.",
          C: "INCORRECT: Debugger analyzes individual training jobs.",
          D: "INCORRECT: Autopilot automatically builds single models, not orchestrates workflows."
        }
      },
      {
        id: 62,
        domain: "ML Implementation and Operations",
        question: "A trained model needs to be stored with metadata, versioning, and approval workflow before deployment. Which SageMaker feature provides this?",
        options: [
          { id: "A", text: "SageMaker Model Registry" },
          { id: "B", text: "SageMaker Feature Store" },
          { id: "C", text: "SageMaker Data Wrangler" },
          { id: "D", text: "SageMaker Ground Truth" }
        ],
        correct: "A",
        explanations: {
          A: "CORRECT: Model Registry provides model versioning, metadata storage, and approval workflows (Pending → Approved → Rejected) for production deployment governance.",
          B: "INCORRECT: Feature Store manages features, not trained models.",
          C: "INCORRECT: Data Wrangler prepares data, doesn't store models.",
          D: "INCORRECT: Ground Truth is for data labeling."
        }
      },
      {
        id: 63,
        domain: "ML Implementation and Operations",
        question: "Training a large deep learning model on a single GPU takes too long. The model fits in GPU memory but the dataset is large. Which parallelism strategy should be used?",
        options: [
          { id: "A", text: "Model parallelism" },
          { id: "B", text: "Data parallelism" },
          { id: "C", text: "Pipeline parallelism" },
          { id: "D", text: "Sequential processing" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Model parallelism is for models too large for one GPU.",
          B: "CORRECT: Data parallelism distributes batches across multiple GPUs, each with a copy of the model. Gradients are synchronized. Speeds training when model fits in memory but data is large.",
          C: "INCORRECT: Pipeline parallelism is a form of model parallelism for very large models.",
          D: "INCORRECT: Sequential processing doesn't use parallelism."
        }
      },
      {
        id: 64,
        domain: "ML Implementation and Operations",
        question: "A production ML endpoint needs to validate that predictions fall within business-acceptable ranges before returning to users. What should be implemented?",
        options: [
          { id: "A", text: "Input validation in the client" },
          { id: "B", text: "SageMaker inference pipeline with post-processing container" },
          { id: "C", text: "SageMaker Model Monitor" },
          { id: "D", text: "Lambda function before the endpoint" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Client-side validation doesn't control what the model returns.",
          B: "CORRECT: Inference pipelines can include post-processing containers that validate, constrain, or transform model outputs before returning to users.",
          C: "INCORRECT: Model Monitor detects drift but doesn't modify predictions in real-time.",
          D: "INCORRECT: Lambda before the endpoint validates inputs, not outputs."
        }
      },
      {
        id: 65,
        domain: "ML Implementation and Operations",
        question: "A company wants to ensure their ML model doesn't discriminate against protected groups. Which SageMaker feature should be used to detect bias before deployment?",
        options: [
          { id: "A", text: "SageMaker Debugger" },
          { id: "B", text: "SageMaker Clarify" },
          { id: "C", text: "SageMaker Model Monitor" },
          { id: "D", text: "SageMaker Experiments" }
        ],
        correct: "B",
        explanations: {
          A: "INCORRECT: Debugger analyzes training issues, not fairness.",
          B: "CORRECT: SageMaker Clarify provides pre-training bias detection (analyzing data) and post-training bias detection (analyzing predictions) across protected attributes.",
          C: "INCORRECT: Model Monitor detects drift in production but Clarify is for initial bias analysis.",
          D: "INCORRECT: Experiments tracks training runs, not bias analysis."
        }
      }
    ]
  }
};

const formatTime = (seconds) => {
  const mins = Math.floor(seconds / 60);
  const secs = seconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const formatDate = () => {
  return new Date().toLocaleString();
};

export default function AWSMLQuiz() {
  // Profile and persistence state
  const [activeProfile, setActiveProfileState] = useState(() => getActiveProfile());
  const [currentAttemptId, setCurrentAttemptId] = useState(null);
  const [viewingAttempt, setViewingAttempt] = useState(null); // For viewing past attempts
  
  // Screen and test state
  const [screen, setScreen] = useState(() => getActiveProfile() ? 'dashboard' : 'profile'); // 'profile', 'dashboard', 'start', 'test', 'results', 'review'
  const [selectedTest, setSelectedTest] = useState(null);
  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [answers, setAnswers] = useState({});
  const [bookmarks, setBookmarks] = useState([]);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [finalTime, setFinalTime] = useState(0);
  const timerRef = useRef(null);
  const autoSaveRef = useRef(null);

  const questions = selectedTest ? testBank[selectedTest].questions : [];

  // Timer effect
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

  // Auto-save progress effect
  const saveProgress = useCallback(() => {
    if (activeProfile && currentAttemptId && screen === 'test') {
      updateAttemptProgress(activeProfile.id, currentAttemptId, {
        answers,
        bookmarks,
        currentQuestion,
        elapsedTime
      });
    }
  }, [activeProfile, currentAttemptId, screen, answers, bookmarks, currentQuestion, elapsedTime]);

  // Save on answer/bookmark change
  useEffect(() => {
    if (screen === 'test' && currentAttemptId) {
      saveProgress();
    }
  }, [answers, bookmarks, currentQuestion]);

  // Auto-save every 30 seconds
  useEffect(() => {
    if (screen === 'test' && currentAttemptId) {
      autoSaveRef.current = setInterval(saveProgress, 30000);
    }
    return () => {
      if (autoSaveRef.current) clearInterval(autoSaveRef.current);
    };
  }, [screen, currentAttemptId, saveProgress]);

  // Save on page unload
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (screen === 'test' && currentAttemptId) {
        saveProgress();
      }
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [screen, currentAttemptId, saveProgress]);

  // Profile handlers
  const handleProfileSelected = (profile) => {
    setActiveProfileState(profile);
    setScreen('dashboard');
  };

  const handleSwitchProfile = () => {
    setActiveProfileState(null);
    setScreen('profile');
    setSelectedTest(null);
    setCurrentAttemptId(null);
    setViewingAttempt(null);
  };

  // Test selection handlers
  const selectTest = (testId) => {
    setSelectedTest(testId);
    setViewingAttempt(null);
    setScreen('start');
  };

  const startTest = () => {
    // Create a new attempt
    const attempt = createAttempt(activeProfile.id, selectedTest);
    setCurrentAttemptId(attempt.id);
    setScreen('test');
    setCurrentQuestion(0);
    setAnswers({});
    setBookmarks([]);
    setElapsedTime(0);
  };

  // Resume an in-progress attempt
  const handleResumeAttempt = (testId, attempt) => {
    setSelectedTest(testId);
    setCurrentAttemptId(attempt.id);
    setAnswers(attempt.answers || {});
    setBookmarks(attempt.bookmarks || []);
    setCurrentQuestion(attempt.currentQuestion || 0);
    setElapsedTime(attempt.elapsedTime || 0);
    setViewingAttempt(null);
    setScreen('test');
  };

  // View past attempt results
  const handleViewResults = (testId, attempt) => {
    setSelectedTest(testId);
    setViewingAttempt(attempt);
    setAnswers(attempt.answers || {});
    setBookmarks(attempt.bookmarks || []);
    setFinalTime(attempt.elapsedTime);
    setScreen('results');
  };

  const handleAnswer = (questionId, answerId) => {
    setAnswers(prev => ({ ...prev, [questionId]: answerId }));
  };

  const toggleBookmark = (questionId) => {
    setBookmarks(prev => {
      if (prev.includes(questionId)) {
        return prev.filter(id => id !== questionId);
      } else {
        return [...prev, questionId];
      }
    });
  };

  // Toggle bookmark in review mode (for completed attempts)
  const toggleBookmarkInReview = (questionId) => {
    const newBookmarks = bookmarks.includes(questionId)
      ? bookmarks.filter(id => id !== questionId)
      : [...bookmarks, questionId];
    
    setBookmarks(newBookmarks);
    
    // Save to storage if viewing a past attempt
    if (viewingAttempt && activeProfile) {
      updateAttemptBookmarks(activeProfile.id, viewingAttempt.id, newBookmarks);
    }
  };

  const finishTest = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    if (autoSaveRef.current) clearInterval(autoSaveRef.current);
    
    const finalScore = calculateScore();
    const finalPercentage = Math.round((finalScore / questions.length) * 100);
    const didPass = finalPercentage >= 75;
    
    // Save completed attempt
    if (activeProfile && currentAttemptId) {
      completeAttempt(activeProfile.id, currentAttemptId, {
        score: finalScore,
        percentage: finalPercentage,
        passed: didPass,
        finalTime: elapsedTime
      });
    }
    
    setFinalTime(elapsedTime);
    setViewingAttempt(null); // We're viewing current attempt, not a past one
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

  const startReviewBookmarked = () => {
    // Find the first bookmarked question
    const firstBookmarkedIdx = questions.findIndex(q => bookmarks.includes(q.id));
    setCurrentQuestion(firstBookmarkedIdx >= 0 ? firstBookmarkedIdx : 0);
    setScreen('review');
  };

  const restartTest = () => {
    setScreen('start');
    setAnswers({});
    setBookmarks([]);
    setElapsedTime(0);
    setFinalTime(0);
    setCurrentQuestion(0);
    setCurrentAttemptId(null);
    setViewingAttempt(null);
  };

  const goToTestSelect = () => {
    setScreen('dashboard');
    setSelectedTest(null);
    setAnswers({});
    setBookmarks([]);
    setElapsedTime(0);
    setFinalTime(0);
    setCurrentQuestion(0);
    setCurrentAttemptId(null);
    setViewingAttempt(null);
  };

  // Export state for showing export options
  const [showExportMenu, setShowExportMenu] = useState(false);

  // Get the attempt data for export (either viewing past attempt or current completed one)
  const getExportAttempt = () => {
    if (viewingAttempt) {
      return viewingAttempt;
    }
    // Build attempt object from current state
    return {
      answers,
      score,
      percentage,
      passed,
      elapsedTime: finalTime,
      startedAt: new Date().toISOString(),
      completedAt: new Date().toISOString()
    };
  };

  const handleExportJSON = () => {
    const attempt = getExportAttempt();
    exportAsJSON(attempt, testBank[selectedTest], questions);
    setShowExportMenu(false);
  };

  const handleExportCSV = () => {
    const attempt = getExportAttempt();
    exportAsCSV(attempt, testBank[selectedTest], questions);
    setShowExportMenu(false);
  };

  const handleExportHTML = () => {
    const attempt = getExportAttempt();
    exportAsHTML(attempt, testBank[selectedTest], questions, domainScores);
    setShowExportMenu(false);
  };

  // Calculate derived values
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

  // Profile Selection Screen
  if (screen === 'profile') {
    return <ProfileSelect onProfileSelected={handleProfileSelected} />;
  }

  // Study Guide Screen
  if (screen === 'studyguide') {
    return <StudyGuide onBack={() => setScreen('dashboard')} />;
  }

  // Test Dashboard Screen
  if (screen === 'dashboard') {
    return (
      <TestDashboard
        profile={activeProfile}
        testBank={testBank}
        onStartTest={selectTest}
        onResumeAttempt={handleResumeAttempt}
        onViewResults={handleViewResults}
        onSwitchProfile={handleSwitchProfile}
        onOpenStudyGuide={() => setScreen('studyguide')}
      />
    );
  }

  // Test Selection Screen (legacy - now redirects to dashboard)
  if (screen === 'select') {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4 sm:p-6">
        <div className="max-w-2xl w-full">
          <div className="text-center mb-6 sm:mb-8">
            <div className="inline-flex items-center justify-center w-14 h-14 sm:w-20 sm:h-20 bg-gradient-to-br from-orange-500 to-amber-600 rounded-xl sm:rounded-2xl mb-4 sm:mb-6 shadow-lg">
              <svg className="w-7 h-7 sm:w-10 sm:h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h1 className="text-xl sm:text-3xl font-bold text-white mb-2">AWS ML Specialty</h1>
            <p className="text-slate-400 text-sm sm:text-lg">Select a Practice Exam</p>
          </div>

          <div className="space-y-3 sm:space-y-4">
            {Object.entries(testBank).map(([testId, test]) => (
              <button
                key={testId}
                onClick={() => selectTest(testId)}
                className="w-full bg-slate-800 rounded-xl sm:rounded-2xl p-4 sm:p-6 border border-slate-700 hover:border-amber-500 transition-all duration-200 text-left group"
              >
                <div className="flex items-center justify-between">
                  <div className="min-w-0 flex-1">
                    <h2 className="text-base sm:text-xl font-semibold text-white group-hover:text-amber-500 transition-colors">
                      {test.name}
                    </h2>
                    <p className="text-slate-400 text-sm sm:text-base mt-1 truncate">{test.description}</p>
                    <p className="text-slate-500 text-xs sm:text-sm mt-1 sm:mt-2">{test.questions.length} questions</p>
                  </div>
                  <svg className="w-5 h-5 sm:w-6 sm:h-6 text-slate-500 group-hover:text-amber-500 transition-colors flex-shrink-0 ml-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Start Screen
  if (screen === 'start') {
    return (
      <div className="min-h-screen bg-slate-900 flex items-center justify-center p-4 sm:p-6">
        <div className="max-w-2xl w-full bg-slate-800 rounded-xl sm:rounded-2xl shadow-2xl p-4 sm:p-8 border border-slate-700">
          <button
            onClick={goToTestSelect}
            className="flex items-center text-slate-400 hover:text-white mb-4 sm:mb-6 transition-colors text-sm sm:text-base"
          >
            <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back
          </button>

          <div className="text-center mb-6 sm:mb-8">
            <div className="inline-flex items-center justify-center w-14 h-14 sm:w-20 sm:h-20 bg-gradient-to-br from-orange-500 to-amber-600 rounded-xl sm:rounded-2xl mb-4 sm:mb-6 shadow-lg">
              <svg className="w-7 h-7 sm:w-10 sm:h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <h1 className="text-xl sm:text-3xl font-bold text-white mb-2">{testBank[selectedTest].name}</h1>
            <p className="text-slate-400 text-sm sm:text-lg">{testBank[selectedTest].description}</p>
          </div>
          
          <div className="bg-slate-700/50 rounded-lg sm:rounded-xl p-4 sm:p-6 mb-6 sm:mb-8">
            <h2 className="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">Exam Overview</h2>
            <div className="grid grid-cols-2 gap-3 sm:gap-4 text-xs sm:text-sm">
              <div className="flex items-center text-slate-300">
                <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-amber-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                30 Questions
              </div>
              <div className="flex items-center text-slate-300">
                <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-amber-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                75% to Pass
              </div>
              <div className="flex items-center text-slate-300">
                <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-amber-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                Timed
              </div>
              <div className="flex items-center text-slate-300">
                <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2 text-amber-500 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                </svg>
                Free Nav
              </div>
            </div>
          </div>

          <div className="bg-slate-700/50 rounded-lg sm:rounded-xl p-4 sm:p-6 mb-6 sm:mb-8">
            <h2 className="text-base sm:text-lg font-semibold text-white mb-2 sm:mb-3">Domain Coverage</h2>
            <div className="space-y-1.5 sm:space-y-2 text-xs sm:text-sm">
              <div className="flex justify-between text-slate-300">
                <span>Data Engineering</span>
                <span className="text-amber-500">6 (20%)</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Exploratory Data Analysis</span>
                <span className="text-amber-500">7 (24%)</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span>Modeling</span>
                <span className="text-amber-500">11 (36%)</span>
              </div>
              <div className="flex justify-between text-slate-300">
                <span className="truncate mr-2">ML Implementation & Ops</span>
                <span className="text-amber-500 flex-shrink-0">6 (20%)</span>
              </div>
            </div>
          </div>

          <button
            onClick={startTest}
            className="w-full py-3 sm:py-4 bg-gradient-to-r from-orange-500 to-amber-600 text-white font-semibold rounded-xl hover:from-orange-600 hover:to-amber-700 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 text-sm sm:text-base"
          >
            Start Exam
          </button>
        </div>
      </div>
    );
  }

  // Test Screen
  if (screen === 'test') {
    const answeredCount = Object.keys(answers).length;
    
    return (
      <div className="min-h-screen bg-slate-900 p-3 sm:p-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="bg-slate-800 rounded-xl p-3 sm:p-4 mb-4 sm:mb-6 border border-slate-700">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-3 sm:gap-6 text-xs sm:text-sm">
                <div className="text-slate-400">
                  Q <span className="text-white font-semibold">{currentQuestion + 1}</span>/{questions.length}
                </div>
                <div className="text-slate-400">
                  <span className="text-amber-500 font-semibold">{answeredCount}</span> answered
                </div>
              </div>
              <div className="flex items-center space-x-2 bg-slate-700 px-3 py-1.5 sm:px-4 sm:py-2 rounded-lg">
                <svg className="w-4 h-4 sm:w-5 sm:h-5 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-white font-mono font-semibold text-sm sm:text-base">{formatTime(elapsedTime)}</span>
              </div>
            </div>
          </div>

          {/* Progress bar */}
          <div className="h-1.5 sm:h-2 bg-slate-700 rounded-full mb-4 sm:mb-6 overflow-hidden">
            <div 
              className="h-full bg-gradient-to-r from-orange-500 to-amber-500 transition-all duration-300"
              style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
            />
          </div>

          {/* Question Card */}
          <div className="bg-slate-800 rounded-xl sm:rounded-2xl p-4 sm:p-8 mb-4 sm:mb-6 border border-slate-700">
            <div className="flex items-center justify-between mb-3 sm:mb-4">
              <div className="inline-block px-2 py-1 sm:px-3 bg-amber-500/20 text-amber-500 text-xs sm:text-sm font-medium rounded-full">
                {q.domain}
              </div>
              <button
                onClick={() => toggleBookmark(q.id)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 sm:px-3 sm:py-2 rounded-lg transition-all text-xs sm:text-sm font-medium ${
                  bookmarks.includes(q.id)
                    ? 'bg-yellow-500/20 text-yellow-500 hover:bg-yellow-500/30'
                    : 'bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-slate-300'
                }`}
                title={bookmarks.includes(q.id) ? 'Remove bookmark' : 'Bookmark for review'}
              >
                <svg 
                  className="w-4 h-4 sm:w-5 sm:h-5" 
                  fill={bookmarks.includes(q.id) ? 'currentColor' : 'none'} 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                </svg>
                <span className="hidden sm:inline">{bookmarks.includes(q.id) ? 'Bookmarked' : 'Bookmark'}</span>
              </button>
            </div>
            <h2 className="text-base sm:text-xl text-white mb-6 sm:mb-8 leading-relaxed">{q.question}</h2>
            
            <div className="space-y-2 sm:space-y-3">
              {q.options.map(option => (
                <label
                  key={option.id}
                  className={`flex items-start p-3 sm:p-4 rounded-lg sm:rounded-xl cursor-pointer transition-all duration-200 border-2 ${
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
                  <div className={`flex-shrink-0 w-7 h-7 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center font-semibold mr-3 sm:mr-4 text-sm sm:text-base ${
                    answers[q.id] === option.id
                      ? 'bg-amber-500 text-white'
                      : 'bg-slate-600 text-slate-300'
                  }`}>
                    {option.id}
                  </div>
                  <span className="text-slate-200 pt-0.5 sm:pt-1 text-sm sm:text-base">{option.text}</span>
                </label>
              ))}
            </div>
          </div>

          {/* Navigation */}
          <div className="flex items-center justify-between gap-2">
            <button
              onClick={() => setCurrentQuestion(prev => Math.max(0, prev - 1))}
              disabled={currentQuestion === 0}
              className="flex items-center px-3 sm:px-6 py-2.5 sm:py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-sm sm:text-base"
            >
              <svg className="w-5 h-5 sm:mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              <span className="hidden sm:inline">Previous</span>
            </button>

            <div className="hidden sm:flex items-center space-x-1.5 overflow-x-auto max-w-[50%] py-2">
              {questions.map((question, idx) => {
                const isBookmarked = bookmarks.includes(question.id);
                const isAnswered = answers[question.id];
                return (
                  <button
                    key={idx}
                    onClick={() => setCurrentQuestion(idx)}
                    className={`w-2.5 h-2.5 rounded-full transition-all flex-shrink-0 ${
                      idx === currentQuestion
                        ? 'bg-amber-500 scale-125'
                        : isAnswered
                        ? 'bg-green-500'
                        : 'bg-slate-600 hover:bg-slate-500'
                    } ${isBookmarked ? 'ring-2 ring-yellow-500 ring-offset-1 ring-offset-slate-900' : ''}`}
                    title={isBookmarked ? 'Bookmarked' : ''}
                  />
                );
              })}
            </div>
            
            {/* Mobile question indicator with bookmark count */}
            <div className="sm:hidden flex items-center gap-2 text-slate-400 text-sm">
              <span>{currentQuestion + 1} / {questions.length}</span>
              {bookmarks.length > 0 && (
                <span className="flex items-center text-yellow-500">
                  <svg className="w-3.5 h-3.5 mr-0.5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                  </svg>
                  {bookmarks.length}
                </span>
              )}
            </div>

            {currentQuestion === questions.length - 1 ? (
              <button
                onClick={finishTest}
                className="flex items-center px-3 sm:px-4 py-2.5 sm:py-3 bg-gradient-to-r from-green-500 to-emerald-600 text-white font-semibold rounded-xl hover:from-green-600 hover:to-emerald-700 transition-all shadow-lg whitespace-nowrap text-sm"
              >
                <span className="hidden sm:inline">Finish</span>
                <span className="sm:hidden">Done</span>
                <svg className="w-5 h-5 sm:ml-1 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </button>
            ) : (
              <button
                onClick={() => setCurrentQuestion(prev => Math.min(questions.length - 1, prev + 1))}
                className="flex items-center px-3 sm:px-6 py-2.5 sm:py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 transition-all text-sm sm:text-base"
              >
                <span className="hidden sm:inline">Next</span>
                <svg className="w-5 h-5 sm:ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </button>
            )}
          </div>
        </div>
      </div>
    );
  }

  // Results Screen
  if (screen === 'results') {
    return (
      <div className="min-h-screen bg-slate-900 p-4 sm:p-6">
        <div className="max-w-2xl mx-auto">
          <div className="bg-slate-800 rounded-xl sm:rounded-2xl p-4 sm:p-8 border border-slate-700 text-center mb-4 sm:mb-6">
            <div className={`inline-flex items-center justify-center w-16 h-16 sm:w-24 sm:h-24 rounded-full mb-4 sm:mb-6 ${
              passed ? 'bg-green-500/20' : 'bg-red-500/20'
            }`}>
              {passed ? (
                <svg className="w-8 h-8 sm:w-12 sm:h-12 text-green-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              ) : (
                <svg className="w-8 h-8 sm:w-12 sm:h-12 text-red-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              )}
            </div>
            
            <h1 className={`text-2xl sm:text-3xl font-bold mb-2 ${passed ? 'text-green-500' : 'text-red-500'}`}>
              {passed ? 'PASSED' : 'NOT PASSED'}
            </h1>
            <p className="text-slate-400 text-sm sm:text-base mb-6 sm:mb-8">{testBank[selectedTest].name} Complete</p>

            <div className="grid grid-cols-3 gap-2 sm:gap-4 mb-6 sm:mb-8">
              <div className="bg-slate-700/50 rounded-lg sm:rounded-xl p-2 sm:p-4">
                <div className="text-xl sm:text-3xl font-bold text-white">{score}/{questions.length}</div>
                <div className="text-slate-400 text-xs sm:text-sm">Correct</div>
              </div>
              <div className="bg-slate-700/50 rounded-lg sm:rounded-xl p-2 sm:p-4">
                <div className={`text-xl sm:text-3xl font-bold ${passed ? 'text-green-500' : 'text-red-500'}`}>{percentage}%</div>
                <div className="text-slate-400 text-xs sm:text-sm">Score</div>
              </div>
              <div className="bg-slate-700/50 rounded-lg sm:rounded-xl p-2 sm:p-4">
                <div className="text-xl sm:text-3xl font-bold text-amber-500">{formatTime(finalTime)}</div>
                <div className="text-slate-400 text-xs sm:text-sm">Time</div>
              </div>
            </div>

            <div className="bg-slate-700/50 rounded-lg sm:rounded-xl p-4 sm:p-6 text-left">
              <h2 className="text-base sm:text-lg font-semibold text-white mb-3 sm:mb-4">Domain Breakdown</h2>
              <div className="space-y-2 sm:space-y-3">
                {Object.entries(domainScores).map(([domain, scores]) => {
                  const domainPct = Math.round((scores.correct / scores.total) * 100);
                  return (
                    <div key={domain}>
                      <div className="flex flex-col sm:flex-row sm:justify-between text-xs sm:text-sm mb-1 gap-0.5">
                        <span className="text-slate-300 truncate">{domain}</span>
                        <span className={`${domainPct >= 75 ? 'text-green-500' : 'text-red-500'} flex-shrink-0`}>
                          {scores.correct}/{scores.total} ({domainPct}%)
                        </span>
                      </div>
                      <div className="h-1.5 sm:h-2 bg-slate-600 rounded-full overflow-hidden">
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

          {/* Attempt info if viewing past attempt */}
          {viewingAttempt && (
            <div className="bg-slate-700/30 rounded-lg sm:rounded-xl p-3 mb-4 text-center">
              <span className="text-slate-400 text-xs sm:text-sm">
                Attempt from {new Date(viewingAttempt.completedAt).toLocaleDateString('en-US', {
                  month: 'short', day: 'numeric', year: 'numeric', hour: 'numeric', minute: '2-digit'
                })}
              </span>
            </div>
          )}

          {/* Bookmarked questions indicator */}
          {bookmarks.length > 0 && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-3 sm:p-4 mb-4 flex items-center justify-between">
              <div className="flex items-center text-yellow-500">
                <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                </svg>
                <span className="text-sm sm:text-base font-medium">{bookmarks.length} question{bookmarks.length !== 1 ? 's' : ''} bookmarked</span>
              </div>
              <button
                onClick={startReviewBookmarked}
                className="px-3 py-1.5 bg-yellow-500/20 text-yellow-500 text-xs sm:text-sm font-medium rounded-lg hover:bg-yellow-500/30 transition-all"
              >
                Review Bookmarked
              </button>
            </div>
          )}

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4 mb-3 sm:mb-4">
            <button
              onClick={startReview}
              className="py-3 sm:py-4 bg-gradient-to-r from-orange-500 to-amber-600 text-white font-semibold rounded-xl hover:from-orange-600 hover:to-amber-700 transition-all shadow-lg text-sm sm:text-base"
            >
              Review All Answers
            </button>
            <div className="relative">
              <button
                onClick={() => setShowExportMenu(!showExportMenu)}
                className="w-full py-3 sm:py-4 bg-slate-700 text-white font-semibold rounded-xl hover:bg-slate-600 transition-all flex items-center justify-center text-sm sm:text-base"
              >
                <svg className="w-4 h-4 sm:w-5 sm:h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                Export
                <svg className={`w-4 h-4 ml-2 transition-transform ${showExportMenu ? 'rotate-180' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
              {showExportMenu && (
                <div className="absolute bottom-full left-0 right-0 mb-2 bg-slate-700 rounded-xl border border-slate-600 overflow-hidden shadow-xl z-10">
                  <button
                    onClick={handleExportHTML}
                    className="w-full px-4 py-3 text-left text-white hover:bg-slate-600 transition-colors flex items-center"
                  >
                    <svg className="w-5 h-5 mr-3 text-amber-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    <div>
                      <div className="font-medium">View Report</div>
                      <div className="text-slate-400 text-xs">Opens printable HTML report</div>
                    </div>
                  </button>
                  <button
                    onClick={handleExportJSON}
                    className="w-full px-4 py-3 text-left text-white hover:bg-slate-600 transition-colors flex items-center border-t border-slate-600"
                  >
                    <svg className="w-5 h-5 mr-3 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                    </svg>
                    <div>
                      <div className="font-medium">Download JSON</div>
                      <div className="text-slate-400 text-xs">Full data export</div>
                    </div>
                  </button>
                  <button
                    onClick={handleExportCSV}
                    className="w-full px-4 py-3 text-left text-white hover:bg-slate-600 transition-colors flex items-center border-t border-slate-600"
                  >
                    <svg className="w-5 h-5 mr-3 text-green-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                    <div>
                      <div className="font-medium">Download CSV</div>
                      <div className="text-slate-400 text-xs">Open in Excel/Sheets</div>
                    </div>
                  </button>
                </div>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
            <button
              onClick={restartTest}
              className="py-3 sm:py-4 bg-slate-700 text-white font-semibold rounded-xl hover:bg-slate-600 transition-all text-sm sm:text-base"
            >
              {viewingAttempt ? 'New Attempt' : 'Retake Exam'}
            </button>
            <button
              onClick={goToTestSelect}
              className="py-3 sm:py-4 bg-slate-700 text-white font-semibold rounded-xl hover:bg-slate-600 transition-all text-sm sm:text-base"
            >
              Dashboard
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Review Screen
  if (screen === 'review') {
    const userAnswer = answers[q.id];
    const isCorrect = userAnswer === q.correct;
    const isBookmarked = bookmarks.includes(q.id);
    
    return (
      <div className="min-h-screen bg-slate-900 p-3 sm:p-6">
        <div className="max-w-4xl mx-auto">
          {/* Header */}
          <div className="bg-slate-800 rounded-xl p-3 sm:p-4 mb-4 sm:mb-6 border border-slate-700">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2 sm:gap-4">
                <span className="text-slate-400 text-xs sm:text-sm">
                  Q <span className="text-white font-semibold">{currentQuestion + 1}</span>/{questions.length}
                </span>
                <span className={`px-2 sm:px-3 py-0.5 sm:py-1 rounded-full text-xs sm:text-sm font-medium ${
                  isCorrect ? 'bg-green-500/20 text-green-500' : 'bg-red-500/20 text-red-500'
                }`}>
                  {isCorrect ? 'Correct' : 'Incorrect'}
                </span>
                {isBookmarked && (
                  <span className="flex items-center px-2 sm:px-3 py-0.5 sm:py-1 rounded-full text-xs sm:text-sm font-medium bg-yellow-500/20 text-yellow-500">
                    <svg className="w-3.5 h-3.5 mr-1" fill="currentColor" viewBox="0 0 24 24">
                      <path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                    </svg>
                    <span className="hidden sm:inline">Bookmarked</span>
                  </span>
                )}
              </div>
              <button
                onClick={() => setScreen('results')}
                className="text-slate-400 hover:text-white transition-colors text-xs sm:text-sm"
              >
                Back to Results
              </button>
            </div>
          </div>

          {/* Question Card */}
          <div className="bg-slate-800 rounded-xl sm:rounded-2xl p-4 sm:p-8 mb-4 sm:mb-6 border border-slate-700">
            <div className="flex items-center justify-between mb-3 sm:mb-4">
              <div className="inline-block px-2 sm:px-3 py-1 bg-amber-500/20 text-amber-500 text-xs sm:text-sm font-medium rounded-full">
                {q.domain}
              </div>
              <button
                onClick={() => toggleBookmarkInReview(q.id)}
                className={`flex items-center gap-1.5 px-2.5 py-1.5 sm:px-3 sm:py-2 rounded-lg transition-all text-xs sm:text-sm font-medium ${
                  isBookmarked
                    ? 'bg-yellow-500/20 text-yellow-500 hover:bg-yellow-500/30'
                    : 'bg-slate-700 text-slate-400 hover:bg-slate-600 hover:text-slate-300'
                }`}
                title={isBookmarked ? 'Remove bookmark' : 'Add bookmark'}
              >
                <svg 
                  className="w-4 h-4 sm:w-5 sm:h-5" 
                  fill={isBookmarked ? 'currentColor' : 'none'} 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                </svg>
                <span className="hidden sm:inline">{isBookmarked ? 'Bookmarked' : 'Bookmark'}</span>
              </button>
            </div>
            <h2 className="text-base sm:text-xl text-white mb-6 sm:mb-8 leading-relaxed">{q.question}</h2>
            
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
                    className={`p-3 sm:p-4 rounded-lg sm:rounded-xl border-2 ${borderColor} ${bgColor}`}
                  >
                    <div className="flex items-start mb-2 sm:mb-3">
                      <div className={`flex-shrink-0 w-7 h-7 sm:w-8 sm:h-8 rounded-lg flex items-center justify-center font-semibold mr-3 sm:mr-4 text-sm sm:text-base ${
                        isThisCorrect
                          ? 'bg-green-500 text-white'
                          : wasSelected
                          ? 'bg-red-500 text-white'
                          : 'bg-slate-600 text-slate-300'
                      }`}>
                        {option.id}
                      </div>
                      <div className="flex-1 min-w-0">
                        <span className="text-slate-200 text-sm sm:text-base">{option.text}</span>
                        {wasSelected && (
                          <span className="ml-1 sm:ml-2 text-xs sm:text-sm text-slate-400">(Yours)</span>
                        )}
                        {isThisCorrect && (
                          <span className="ml-1 sm:ml-2 text-xs sm:text-sm text-green-500">(Correct)</span>
                        )}
                      </div>
                    </div>
                    <div className={`ml-10 sm:ml-12 text-xs sm:text-sm p-2 sm:p-3 rounded-lg ${
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
          <div className="flex items-center justify-between gap-2">
            <button
              onClick={() => setCurrentQuestion(prev => Math.max(0, prev - 1))}
              disabled={currentQuestion === 0}
              className="flex items-center px-3 sm:px-6 py-2.5 sm:py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-sm sm:text-base"
            >
              <svg className="w-5 h-5 sm:mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              <span className="hidden sm:inline">Previous</span>
            </button>

            <div className="hidden sm:flex items-center space-x-1.5 overflow-x-auto max-w-[50%] py-2">
              {questions.map((question, idx) => {
                const wasCorrect = answers[question.id] === question.correct;
                const isBookmarked = bookmarks.includes(question.id);
                return (
                  <button
                    key={idx}
                    onClick={() => setCurrentQuestion(idx)}
                    className={`w-2.5 h-2.5 rounded-full transition-all flex-shrink-0 ${
                      idx === currentQuestion
                        ? 'scale-125 ring-2 ring-white ring-offset-2 ring-offset-slate-900'
                        : ''
                    } ${wasCorrect ? 'bg-green-500' : 'bg-red-500'} ${isBookmarked && idx !== currentQuestion ? 'ring-2 ring-yellow-500 ring-offset-1 ring-offset-slate-900' : ''}`}
                    title={isBookmarked ? 'Bookmarked' : ''}
                  />
                );
              })}
            </div>
            
            {/* Mobile question indicator with bookmark count */}
            <div className="sm:hidden flex items-center gap-2 text-slate-400 text-sm">
              <span>{currentQuestion + 1} / {questions.length}</span>
              {bookmarks.length > 0 && (
                <span className="flex items-center text-yellow-500">
                  <svg className="w-3.5 h-3.5 mr-0.5" fill="currentColor" viewBox="0 0 24 24">
                    <path d="M5 5a2 2 0 012-2h10a2 2 0 012 2v16l-7-3.5L5 21V5z" />
                  </svg>
                  {bookmarks.length}
                </span>
              )}
            </div>

            <button
              onClick={() => setCurrentQuestion(prev => Math.min(questions.length - 1, prev + 1))}
              disabled={currentQuestion === questions.length - 1}
              className="flex items-center px-3 sm:px-6 py-2.5 sm:py-3 bg-slate-700 text-white rounded-xl hover:bg-slate-600 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-sm sm:text-base"
            >
              <span className="hidden sm:inline">Next</span>
              <svg className="w-5 h-5 sm:ml-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
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
