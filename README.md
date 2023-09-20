# Credit_Card_Fraude_detection_Deep_learning
# Table of content
Introduction

Problem Statement

Business Problem Overview

Objective

Data Overview

Data Source

Data Dictionary

Understanding and Defining Fraud

Types of Credit Card Fraud

Project Pipeline

Data Understanding

Exploratory Data Analytics (EDA)

Data Preprocessing

Train/Test Split

Model Building and Hyperparameter Tuning

Model Evaluation

Technologies Used

Steps to Reproduce

Results and Discussion

Future Work

References

Acknowledgements

# Introduction
Problem Statement
The aim of this project is to predict fraudulent credit card transactions using machine learning algorithms.

# Business Problem Overview
Banking fraud is a significant issue that affects both financial institutions and customers. As per the Nilson Report, banking fraud could amount to $30 billion worldwide by 2020.

# Objective
To develop a predictive model that can accurately detect fraudulent transactions, thereby helping in fraud prevention.

# Data Overview
Data Source
The dataset used in this project is sourced from Kaggle. It contains 2,84,807 transactions, out of which 492 are fraudulent.

# Data Dictionary
Time: Seconds elapsed between transactions

Amount: Transaction amount

V1-V28: Anonymized features

Class: 1 for fraudulent transactions and 0 for legitimate ones

# Understanding and Defining Fraud
Types of Credit Card Fraud

Skimming

Manipulation of genuine cards

Creation of counterfeit cards

Stealing/loss of credit cards

Fraudulent telemarketing

# Project Pipeline
Data Understanding
The first step involves understanding the dataset's features and structure.

Exploratory Data Analytics (EDA)
EDA involves univariate and bivariate analyses, along with necessary feature transformations.

Data Preprocessing
Handling missing values and encoding categorical variables.

Train/Test Split
The data is split into training and test sets for model evaluation.

Model Building and Hyperparameter Tuning
Various machine learning models are trained, and their hyperparameters are fine-tuned.

Model Evaluation
Models are evaluated based on metrics that consider the dataset's imbalance.

# Technologies Used
Python

Jupyter Notebook

Scikit-learn

Pandas

Matplotlib

# Steps to Reproduce
Clone the Repository: Clone the GitHub repository to your local machine.

Install Dependencies: Install all the Python libraries and dependencies listed in the requirements.txt file.

Data Preparation: Download the dataset from Kaggle and place it in the designated folder.

Run Notebooks: Execute the Jupyter Notebooks in the order specified.

Model Training: Run the script or notebook cell that initiates model training.

Model Evaluation: Evaluate the model using the test dataset.

# Results and Discussion
Best Model
The best-performing models in the project are as follows:
Logistic Regression with both SMOTE and ADASYN balanced data

Hyper-tuned SVM with both SMOTE and ADASYN balanced data (Note: This model is resource-intensive)

Hyper-tuned XGBoost with both SMOTE and ADASYN balanced data (Note: The model has lower sensitivity)

Evaluation Metrics
Accuracy

Sensitivity

Specificity

ROC curve

# Future Work
Data Augmentation: Investigate techniques for augmenting the dataset to improve model performance.

Model Tuning: Fine-tune the hyperparameters of the best-performing model to improve accuracy further.

Real-time Monitoring: Implement the model in a real-time monitoring system for credit card transactions.

User Interface: Develop a user-friendly interface for fraud detection and reporting.

# References
Dataset:
The dataset is sourced from Kaggle. Credit Card Fraud Detection Dataset on Kaggle

Scikit-learn: Scikit-learn Documentation

Pandas: Pandas Documentation

Matplotlib: Matplotlib Documentation

Miscellaneous:
Nilson Report: Information regarding worldwide banking frauds. Nilson Report

# Acknowledgements
Thanks to those who contributed to the project.

