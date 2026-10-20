# MLOps_assignment
# MLOps Assignment — Questions & Answers

![MLOps](https://img.shields.io/badge/Topic-MLOps-blue?style=for-the-badge)
![Level](https://img.shields.io/badge/Level-Basic%20to%20Advanced-success?style=for-the-badge)

## Topics Covered

- MLOps fundamentals and the end-to-end ML lifecycle
- Git, GitHub, version control, experiment tracking, and model registry
- Data collection, data cleaning, training, validation, and testing
- ML pipelines and workflow automation
- Model deployment, APIs, batch prediction, and real-time prediction
- Streamlit, Docker, cloud platforms, and CI/CD
- Logging, error tracking, monitoring, data drift, and model drift
- Model retraining and security in deployed ML systems


## Table of Contents

1. [MLOps fundamentals](#mlops-fundamentals)
2. [Version control with Git and GitHub](#version-control-with-git-and-github)
3. [Data collection and data cleaning](#data-collection-and-data-cleaning)
4. [Model training and data splitting](#model-training-and-data-splitting)
5. [Model validation and testing](#model-validation-and-testing)
6. [Pipelines and workflow automation](#pipelines-and-workflow-automation)
7. [Experiment tracking and model versioning](#experiment-tracking-and-model-versioning)
8. [Model registry](#model-registry)
9. [Model deployment and APIs](#model-deployment-and-apis)
10. [Batch and real-time prediction](#batch-and-real-time-prediction)
11. [Streamlit and cloud deployment](#streamlit-and-cloud-deployment)
12. [Docker and containers](#docker-and-containers)
13. [CI/CD for ML](#cicd-for-ml)
14. [Cloud computing for ML](#cloud-computing-for-ml)
15. [Logging and error tracking](#logging-and-error-tracking)
16. [Model monitoring and drift](#model-monitoring-and-drift)
17. [Model retraining](#model-retraining)
18. [Security in ML deployment](#security-in-ml-deployment)
19. [Additional MLOps concepts](#additional-mlops-concepts)

---

## MLOps fundamentals

### 1. What is a basic MLOps project flow?

A typical flow is collect data → clean data → train → validate → register/version → deploy → monitor → retrain when needed. **Example:** Automate the steps for a churn model.

### 2. What is MLOps?

MLOps (Machine Learning Operations) combines ML development and operations to deploy, monitor, and maintain models reliably. **Example:** Automating a model's training and deployment.

### 3. Why do we need MLOps in Machine Learning projects?

It makes ML workflows repeatable, scalable, easier to monitor, and safer to update. **Example:** Automatically redeploying a validated model.

### 4. What is the difference between ML and MLOps?

Machine Learning builds models; MLOps manages the process of testing, deploying, monitoring, and maintaining them. **Example:** Training a churn model is ML; keeping it reliable in production is MLOps.

### 5. Why is MLOps used in Machine Learning projects?

It makes ML workflows repeatable, scalable, easier to monitor, and safer to update. **Example:** Automatically redeploying a validated model.

---

## Version control with Git and GitHub

### 1. What is Git?

Git is a version-control system that records changes to files over time. **Example:** Track edits to a model-training script.

### 2. What is GitHub?

GitHub is an online platform for hosting Git repositories and collaborating on code. **Example:** Share an ML project repository with a team.

### 3. What is the difference between Git and GitHub?

Git tracks code changes locally; GitHub hosts Git repositories online and supports collaboration. **Example:** Commit code with Git and push it to GitHub.

### 4. What is a commit in Git?

A commit is a saved snapshot of changes in a Git repository. **Example:** Commit a fix to `train.py`.

### 5. What is the use of version control in MLOps?

Git and GitHub help track changes, collaborate, review code, and reproduce project history. **Example:** Restore an earlier working version of a training script.

---

## Data collection and data cleaning

### 1. What is data cleaning in Machine Learning?

Data cleaning detects and corrects missing, duplicate, invalid, or inconsistent data. **Example:** Standardizing dates to YYYY-MM-DD.

### 2. Why is data cleaning important before model training?

Cleaning improves data consistency and reduces errors that can mislead a model. **Example:** Removing duplicated transaction rows.

### 3. What are missing values and how can we handle them?

Missing values are blank or unavailable entries. Handle them by removing suitable rows or filling values using methods such as the median. **Example:** Fill missing ages with the median age.

### 4. What is the difference between duplicate data and inconsistent data?

Duplicates repeat the same record; inconsistent data contains conflicting formats or values. **Example:** 'Pune' and 'PUNE' may need standardization.

### 5. What are outliers and why should we check them carefully?

Outliers are unusually high or low observations. Check whether they are errors or genuine cases before changing them. **Example:** Verify a recorded age of 220.

### 6. What is data collection in Machine Learning?

Data collection is the process of gathering information for analysis and model building. **Example:** Gathering past house prices to train a price predictor.

### 7. Why is data collection important in an ML project?

The model learns from collected data, so poor or insufficient data can produce unreliable predictions. **Example:** Collecting enough labeled fraud cases.

### 8. What are different sources of data collection?

Data can come from databases, files, APIs, sensors, surveys, or application logs. **Example:** Collecting sales records from a CSV file.

### 9. What is the difference between good data and bad data?

This is a process or tool used to make ML systems more reliable, repeatable, and easier to maintain. **Example:** Record the result of each model run so the team can review it.

### 10. What problems can happen during data collection?

Common problems include missing, inaccurate, biased, outdated, or insufficient data. **Example:** A survey that excludes many customer groups.

---

## Model training and data splitting

### 1. Why is model deployment important after model training?

Model training is the process of fitting an algorithm to data so it learns patterns. **Example:** Train a classifier using labeled spam and non-spam emails.

### 2. What is the difference between training data and testing data?

Training data teaches the model; testing data evaluates it on unseen examples. **Example:** Train on 80% of records and test on the remaining 20%.

### 3. What is model training in Machine Learning?

Model training is the process of fitting an algorithm to data so it learns patterns. **Example:** Train a classifier using labeled spam and non-spam emails.

### 4. Why do we split data into training and testing data?

Splitting keeps some data unseen during learning so performance can be checked fairly. **Example:** Use 80% for training and 20% for testing.

### 5. What is the difference between features and target?

Features are the input columns; the target is the value the model predicts. **Example:** House size is a feature and house price is the target.

### 6. What is overfitting in model training?

Overfitting happens when a model learns training details too closely and performs poorly on new data. Compare training and test performance to help detect it. **Example:** 99% training accuracy but 70% test accuracy.

### 7. Why is data quality important during model training?

Accurate, representative data helps the model learn useful patterns and make reliable predictions. **Example:** Correct product prices help train a price predictor.

---

## Model validation and testing

### 1. Why should model validation be included in a CI/CD pipeline?

Model validation checks performance or suitability before final release; validation data helps compare models and settings. **Example:** Compare two models using validation accuracy.

### 2. What is the difference between model testing and model monitoring?

Model testing evaluates a trained model against requirements and unseen data. **Example:** Check whether a churn model meets a target F1-score.

### 3. What is model testing in Machine Learning?

Model testing evaluates a trained model against requirements and unseen data. **Example:** Check whether a churn model meets a target F1-score.

### 4. Why do we test a model after training?

Testing checks how well the trained model performs on unseen data before use. **Example:** Evaluate a classifier on held-out customer records.

### 5. Which evaluation metrics are used for classification and regression models?

Classification metrics include accuracy, precision, recall, and F1-score; regression metrics include MAE, MSE, and RMSE. **Example:** Use recall when missing positive cases is costly.

### 6. What is model validation in Machine Learning?

Model validation checks performance or suitability before final release; validation data helps compare models and settings. **Example:** Compare two models using validation accuracy.

### 7. Why do we use validation data?

Model validation checks performance or suitability before final release; validation data helps compare models and settings. **Example:** Compare two models using validation accuracy.

### 8. What is the difference between validation data and testing data?

Validation data helps choose settings during development; test data gives a final evaluation after choices are made. **Example:** Use validation data to select model depth.

### 9. What is cross-validation?

Cross-validation splits data into several folds and repeatedly trains and evaluates the model to estimate performance more robustly. **Example:** In 5-fold CV, each fold is used for validation once.

### 10. How does validation help in hyperparameter tuning?

Validation scores compare hyperparameter settings and help select the best configuration. **Example:** Compare several tree depths and choose the one with the best validation score.

---

## Pipelines and workflow automation

### 1. What is automation in ML workflow?

Automation means using tools or scripts to run ML workflow tasks with minimal manual work. **Example:** Automatically preprocess data and train a model.

### 2. Why is automation useful in MLOps?

It saves time, improves consistency, and reduces manual mistakes. **Example:** Automatically retrain a model every month.

### 3. How does automation help in model deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 4. What are some challenges of automation in ML projects?

Challenges include changing data, complex dependencies, failures, cost, and debugging automated steps. **Example:** A pipeline fails when a required input file is missing.

### 5. What is a pipeline in Machine Learning?

An ML pipeline is a sequence of connected steps for preparing data, training, evaluating, or deploying a model. **Example:** Clean data → train model → evaluate model.

### 6. Why do we use pipelines in MLOps?

Pipelines make repeated steps consistent, traceable, and easier to automate. **Example:** Run the same cleaning and training steps for each dataset update.

### 7. What is the difference between data pipeline and training pipeline?

A data pipeline collects and prepares data; a training pipeline prepares data, trains a model, evaluates it, and may package it. **Example:** Clean a CSV first, then train a classifier.

### 8. How does a deployment pipeline help in ML projects?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 9. Which tools are commonly used to create ML pipelines?

Tools include scikit-learn Pipelines, Airflow, Kubeflow, and cloud ML workflow services. **Example:** Use Airflow to schedule a training workflow.

---

## Experiment tracking and model versioning

### 1. Why is experiment tracking important in MLOps?

Experiment tracking records what was tried and how each model performed. **Example:** Log parameters and accuracy for each training run.

### 2. What is experiment tracking in MLOps?

Experiment tracking records what was tried and how each model performed. **Example:** Log parameters and accuracy for each training run.

### 3. Why is experiment tracking important in Machine Learning projects?

Experiment tracking records what was tried and how each model performed. **Example:** Log parameters and accuracy for each training run.

### 4. What is the difference between experiment tracking and model versioning?

Experiment tracking records training runs and results; model versioning identifies saved model iterations. **Example:** Compare runs, then version the selected model as v2.

### 5. How does experiment tracking help in selecting the best model?

It makes runs comparable so teams can choose a model using measured results. **Example:** Select the run with the highest validation F1-score.

### 6. What is the difference between model versioning and model registry?

Versioning labels and preserves model iterations; a registry organizes those versions with metadata and lifecycle status. **Example:** Register model v3 as approved for production.

### 7. What is model versioning in MLOps?

Model versioning tracks changes to model artifacts so they can be compared, reproduced, or restored. **Example:** Keep v1 and v2 of a churn model.

### 8. Why is model versioning important in Machine Learning projects?

Model versioning tracks changes to model artifacts so they can be compared, reproduced, or restored. **Example:** Keep v1 and v2 of a churn model.

### 9. What is rollback in model versioning?

Rollback means restoring an earlier known-good model version. **Example:** Switch back to v1 after v2 causes errors.

### 10. How does model versioning help in production deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

---

## Model registry

### 1. What is the role of model registry in an MLOps project?

A model registry is a central place to manage model versions and their metadata. **Example:** Store approved churn-model v2 with its evaluation metrics.

### 2. What is a model registry in MLOps?

A model registry is a central place to manage model versions and their metadata. **Example:** Store approved churn-model v2 with its evaluation metrics.

### 3. Why is model registry important in Machine Learning projects?

A model registry is a central place to manage model versions and their metadata. **Example:** Store approved churn-model v2 with its evaluation metrics.

### 4. What information is stored in a model registry?

A registry can store model versions, metrics, artifacts, stage/status, metadata, and lineage. **Example:** Record model v3, its F1-score, and approval status.

### 5. How does model registry help in safe model deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

---

## Model deployment and APIs

### 1. What is an API?

An API (Application Programming Interface) lets software systems communicate using defined requests and responses. **Example:** A Python app asks a prediction service for a result.

### 2. Why is API used for Machine Learning models?

An API lets other applications send input to a model and receive predictions without accessing its internal code. **Example:** A website requests a churn score.

### 3. What is an endpoint in an API?

An endpoint is a specific URL or route where an API receives requests. **Example:** `/predict` accepts data and returns a prediction.

### 4. What is the difference between GET and POST methods?

This is a process or tool used to make ML systems more reliable, repeatable, and easier to maintain. **Example:** Record the result of each model run so the team can review it.

### 5. How does an API help deploy an ML model?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 6. How does cloud help in ML model deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 7. How does Docker help in model deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 8. What is model deployment in Machine Learning?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 9. What is the difference between API deployment and web app deployment?

An API (Application Programming Interface) lets software systems communicate using defined requests and responses. **Example:** A Python app asks a prediction service for a result.

### 10. Which tools are commonly used for model deployment?

Common choices include FastAPI, Flask, Streamlit, Docker, and cloud platforms. **Example:** Use FastAPI to expose a prediction endpoint.

### 11. How does API help in real-time prediction?

An API accepts input, runs the model, and returns the prediction to the requesting application. **Example:** Send transaction details and receive a fraud score.

### 12. Why do we use Streamlit for ML model deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

---

## Batch and real-time prediction

### 1. What is batch prediction in Machine Learning?

Batch prediction generates predictions for many records in one run rather than responding to each request separately. **Example:** Predict churn risk for all customers each night.

### 2. Why do we use batch prediction?

It efficiently handles large numbers of predictions when instant answers are unnecessary. **Example:** Generate next-day demand forecasts for every store.

### 3. What is the difference between batch prediction and real-time prediction?

Batch prediction processes many records together at a scheduled time; real-time prediction returns an answer for an individual request immediately. **Example:** Nightly sales forecasts versus an instant fraud check.

### 4. How does batch prediction work in an ML project?

The system loads a dataset, predicts in bulk, and saves results for later use. **Example:** Score all customers overnight and save churn probabilities.

### 5. What are some challenges in batch prediction?

Challenges include large data volumes, delayed results, failed jobs, duplicate processing, and resource use. **Example:** A nightly job may fail if the input file is incomplete.

### 6. What is real-time prediction in Machine Learning?

Real-time prediction returns a model result shortly after a request arrives. **Example:** Predict delivery time when a user places an order.

### 7. Why do we use real-time prediction?

It supports decisions that need an immediate result. **Example:** Estimate fraud risk during a card payment.

### 8. What is the difference between real-time prediction and batch prediction?

Batch prediction processes many records together at a scheduled time; real-time prediction returns an answer for an individual request immediately. **Example:** Nightly sales forecasts versus an instant fraud check.

### 9. What are some challenges in real-time prediction systems?

Challenges include low latency, high traffic, service outages, security, and keeping models available. **Example:** A slow endpoint delays a fraud decision.

---

## Streamlit and cloud deployment

### 1. What is Streamlit deployment?

Streamlit deployment publishes a Streamlit app so users can access its interface online. **Example:** Host a churn-prediction dashboard.

### 2. What is the use of \requirements.txt\ in deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 3. Why is GCP Cloud Run useful for deploying Streamlit apps?

Cloud provides scalable storage, computing resources, and deployment services without owning all infrastructure. **Example:** Train a model on a temporary cloud VM.

---

## Docker and containers

### 1. What is Docker?

Docker packages an application and its dependencies into a portable container environment. **Example:** Package a FastAPI model service with its Python libraries.

### 2. Why do we use Docker in ML and MLOps projects?

Docker packages an application and its dependencies into a portable container environment. **Example:** Package a FastAPI model service with its Python libraries.

### 3. What is the difference between Docker image and Docker container?

An image is a packaged template; a container is a running instance of that image. **Example:** One image can start several containers.

### 4. What is a Dockerfile?

A Dockerfile contains instructions for building a Docker image, including the base image, dependencies, and startup command. **Example:** A file installs Python packages and runs app.py.

---

## CI/CD for ML

### 1. What is CI/CD?

CI/CD stands for Continuous Integration and Continuous Delivery/Deployment; it automates code checks and release workflows. **Example:** Test a change and package it through a pipeline.

### 2. What is the difference between Continuous Integration and Continuous Deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 3. Why is CI/CD useful in MLOps?

CI/CD makes software and model changes repeatable, testable, and faster to release. **Example:** Run automated API tests before deployment.

### 4. Which tools are commonly used for CI/CD?

Common tools include GitHub Actions, GitLab CI/CD, Jenkins, and cloud build services. **Example:** Use GitHub Actions to run tests on each push.

---

## Cloud computing for ML

### 1. What is cloud in Machine Learning?

Cloud ML uses internet-accessible computing, storage, and services to build or run ML systems. **Example:** Host a prediction API on a cloud platform.

### 2. Why do we use cloud for ML projects?

Cloud provides scalable storage, computing resources, and deployment services without owning all infrastructure. **Example:** Train a model on a temporary cloud VM.

### 3. What is the difference between cloud storage and cloud compute?

Cloud storage holds files and data; cloud compute provides processing power to run code or models. **Example:** Store a dataset in cloud storage and train on a virtual machine.

### 4. What are some challenges of using cloud for ML projects?

Challenges include cost control, security, network dependence, configuration, and vendor-specific services. **Example:** An oversized GPU instance increases costs.

---

## Logging and error tracking

### 1. What is error tracking in MLOps?

Error tracking captures, groups, and helps diagnose failures in a deployed system. **Example:** Report repeated prediction-time exceptions.

### 2. Why is error tracking important for deployed ML models?

Error tracking captures, groups, and helps diagnose failures in a deployed system. **Example:** Report repeated prediction-time exceptions.

### 3. What is the difference between logging and error tracking?

Logging records general events and diagnostic details; error tracking groups and reports application failures. **Example:** Logs capture a request, while an error tracker groups repeated stack traces.

### 4. How does error tracking help improve production ML systems?

Error tracking captures, groups, and helps diagnose failures in a deployed system. **Example:** Report repeated prediction-time exceptions.

### 5. What is logging in MLOps?

Logging records events, warnings, errors, and useful diagnostic details from an application. **Example:** Log the prediction request time and status code.

### 6. Why is logging important in Machine Learning projects?

Logging records events, warnings, errors, and useful diagnostic details from an application. **Example:** Log the prediction request time and status code.

### 7. What are different logging levels?

Common levels are DEBUG, INFO, WARNING, ERROR, and CRITICAL. **Example:** Use ERROR when a prediction request fails.

### 8. What is the difference between logging and monitoring?

Logging records events; monitoring summarizes system health over time and can trigger alerts. **Example:** Logs record errors while a dashboard tracks error rate.

---

## Model monitoring and drift

### 1. Why is model monitoring needed after deployment?

Monitoring checks prediction quality, latency, errors, and data changes after release. **Example:** Alert when API error rates rise.

### 2. What is data drift in MLOps?

Data drift is a change in the distribution or characteristics of input data over time. **Example:** A model receives a much higher share of mobile users than before.

### 3. Why is data drift important in Machine Learning projects?

Changed inputs can make predictions less reliable and may signal a need to investigate or retrain. **Example:** A shopping model struggles after customer buying habits change.

### 4. What is the difference between data drift and model drift?

Data drift means input-data patterns change; model drift means predictive performance or relationships deteriorate. **Example:** Customer ages shift (data drift), while churn predictions become less accurate (model drift).

### 5. How can we detect data drift?

Compare current input distributions with a baseline using statistics, dashboards, or drift tools. **Example:** Alert when customer income distribution changes sharply.

### 6. How can data drift affect a deployed ML model?

Changed inputs can make predictions less reliable and may signal a need to investigate or retrain. **Example:** A shopping model struggles after customer buying habits change.

### 7. Why is model monitoring important after deployment?

Monitoring checks prediction quality, latency, errors, and data changes after release. **Example:** Alert when API error rates rise.

### 8. What is model drift in MLOps?

Model drift is a change that causes a model's predictions or performance to become less reliable over time. **Example:** A previously accurate demand model performs poorly after buying trends change.

### 9. Why does model drift happen after deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 10. What is the difference between data drift and model drift?

Data drift means input-data patterns change; model drift means predictive performance or relationships deteriorate. **Example:** Customer ages shift (data drift), while churn predictions become less accurate (model drift).

### 11. How can we detect model drift?

Track model metrics against labeled outcomes and monitor relevant input/target relationships. **Example:** Alert when monthly F1-score drops below 0.80.

### 12. How can retraining help handle model drift?

Retrain with recent, representative, labeled data, then validate before replacing the production model. **Example:** Retrain a demand model using the latest sales seasons.

### 13. What is model monitoring in MLOps?

Model monitoring tracks a deployed model's health, inputs, outputs, and performance. **Example:** Track weekly prediction accuracy.

### 14. Why is model monitoring important after deployment?

Monitoring checks prediction quality, latency, errors, and data changes after release. **Example:** Alert when API error rates rise.

### 15. What is data drift?

Data drift is a change in the distribution or characteristics of input data over time. **Example:** A model receives a much higher share of mobile users than before.

### 16. What is model drift?

Model drift is a change that causes a model's predictions or performance to become less reliable over time. **Example:** A previously accurate demand model performs poorly after buying trends change.

### 17. What is model monitoring in MLOps?

Model monitoring tracks a deployed model's health, inputs, outputs, and performance. **Example:** Track weekly prediction accuracy.

### 18. What is model drift and why is it important?

It can reduce prediction reliability as real-world patterns change. **Example:** A fraud model misses new fraud patterns.

### 19. Why is model monitoring important after deployment?

Monitoring checks prediction quality, latency, errors, and data changes after release. **Example:** Alert when API error rates rise.

### 20. What is model drift and how does MLOps help in handling it?

Model drift is a change that causes a model's predictions or performance to become less reliable over time. **Example:** A previously accurate demand model performs poorly after buying trends change.

---

## Model retraining

### 1. How does retraining help improve a deployed ML model?

Retraining fits a model again using newer or improved data to maintain performance. **Example:** Update a sales forecast model with recent sales.

### 2. What is model retraining in MLOps?

Retraining fits a model again using newer or improved data to maintain performance. **Example:** Update a sales forecast model with recent sales.

### 3. Why do we retrain Machine Learning models?

Retraining fits a model again using newer or improved data to maintain performance. **Example:** Update a sales forecast model with recent sales.

### 4. When should a deployed model be retrained?

Retrain when performance drops, data patterns change, or enough new representative data becomes available. **Example:** Retrain after a sustained decline in recall.

### 5. What is the difference between scheduled retraining and trigger-based retraining?

Scheduled retraining runs at fixed intervals; trigger-based retraining starts when a condition is met. **Example:** Retrain monthly versus when drift exceeds a threshold.

### 6. How does MLOps help in model retraining?

Retraining fits a model again using newer or improved data to maintain performance. **Example:** Update a sales forecast model with recent sales.

---

## Security in ML deployment

### 1. What is security in ML deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 2. Why is API authentication important in ML deployment?

Authentication verifies who is making a request and helps prevent unauthorized access. **Example:** Require a valid API token.

### 3. What is the difference between authentication and authorization?

Authentication verifies identity; authorization decides what that identity is allowed to do. **Example:** Login verifies a user, while a role check allows only admins to deploy.

### 4. Why is input validation important for ML APIs?

Input validation checks request fields, types, sizes, and allowed values before processing. **Example:** Reject a negative age if the application requires age to be nonnegative.

### 5. What are some common security risks in deployed ML systems?

Risks include exposed API keys, unauthorized access, unsafe inputs, data leaks, and vulnerable dependencies. **Example:** A public repository accidentally contains an API key.

---

## Additional MLOps concepts

### 1. Which ML workflow steps can be automated?

This is a process or tool used to make ML systems more reliable, repeatable, and easier to maintain. **Example:** Record the result of each model run so the team can review it.

### 2. What are common errors in ML APIs?

An API (Application Programming Interface) lets software systems communicate using defined requests and responses. **Example:** A Python app asks a prediction service for a result.

### 3. What information should be tracked in an ML experiment?

This is a process or tool used to make ML systems more reliable, repeatable, and easier to maintain. **Example:** Record the result of each model run so the team can review it.

### 4. Why are Git and GitHub useful in MLOps projects?

Git and GitHub help track changes, collaborate, review code, and reproduce project history. **Example:** Restore an earlier working version of a training script.

### 5. Why should sensitive user data not be stored directly in logs?

This is a process or tool used to make ML systems more reliable, repeatable, and easier to maintain. **Example:** Record the result of each model run so the team can review it.

### 6. What is overfitting and how can testing help detect it?

Overfitting happens when a model learns training details too closely and performs poorly on new data. Compare training and test performance to help detect it. **Example:** 99% training accuracy but 70% test accuracy.

### 7. What information should be stored with each model version?

This is a process or tool used to make ML systems more reliable, repeatable, and easier to maintain. **Example:** Record the result of each model run so the team can review it.

### 8. Why should a retrained model be validated before deployment?

Model deployment makes a trained model available in a real application or service. **Example:** Deploy a price predictor behind an API.

### 9. What is the difference between local app and deployed app?

This is a process or tool used to make ML systems more reliable, repeatable, and easier to maintain. **Example:** Record the result of each model run so the team can review it.

---

