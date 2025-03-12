#Machine Learning Model Comparison for Text Classification

##Usage
Prepare your datasets (data.csv and validation_data.csv).
Run the notebook to preprocess the data, train the models, and evaluate their performance. You can run the entire pipeline or use specific sections as needed.
The results will be displayed for each model, including accuracy, precision, recall, and F1 score.
The predictions for the validation data will be saved as validation_data_predicted.csv.

##Technologies
Python 3.x
Pandas for data manipulation
scikit-learn for machine learning models and evaluation

##Project Structure
data.csv: Input dataset for training.
validation_data.csv: Input dataset for validation.
validation_data_predicted.csv: Output file containing the validation data with predicted labels.
model_comparison.ipynb: Jupyter notebook containing the code for training and evaluating the models.

##Results
The following models were trained and evaluated:
Naive Bayes: Achieved an accuracy of 91.7%.
Logistic Regression: Achieved an accuracy of 98.6%.
Random Forest: Achieved an accuracy of 99.7%.
You can compare the performance of these models based on metrics like accuracy, precision, recall, and F1 score.