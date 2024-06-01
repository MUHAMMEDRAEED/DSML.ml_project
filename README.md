# DSML.ml_project
Project on Machine Learning

Overview of the Machine Learning Classification Project
Objective

The objective of this project is to create a supervised learning model to classify messages (e.g., spam or ham) using a dataset named 'mail_data.csv'. The project involves preprocessing the text data, training a machine learning model, evaluating its performance, and visualizing the results.
Project Workflow

   1. Load the Dataset: Read the mail_data.csv file into a pandas DataFrame.
   2. Data Preprocessing:
       * Handle missing values.
       * Encode the categorical target variable.
       * Transform text data into numerical format using 'TfidfVectorizer'.
   3. Train-Test Split: Split the data into training and testing sets.
   4. Model Selection and Training: Use a Random Forest classifier to train the model.
   5. Model Evaluation: Evaluate the model's performance using accuracy score and confusion matrix.
   6. Visualization: Visualize the results using a confusion matrix.
