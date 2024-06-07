import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# Load the dataset
df = pd.read_csv('E:/Projects/Intelligent-Question-Tagging-Engine/Data2.csv')
df.head()

# Split the data into training and testing sets
X = df['Questions']
y_category = df['Category']
X_train, X_test, y_category_train, y_category_test = train_test_split(X, y_category, test_size=0.2, random_state=42)

# Define the pipeline for category prediction
category_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', OneVsRestClassifier(LinearSVC(max_iter=10000)))  # Increase max_iter parameter
])

# Define the hyperparameters to tune
parameters = {
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Unigrams or bigrams
    'clf__estimator__C': [1, 10, 100]  # Penalty parameter C of the error term
}

# Perform grid search to find the best hyperparameters
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    grid_search = GridSearchCV(category_pipeline, parameters, cv=5)
    grid_search.fit(X_train, y_category_train)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Get the best performing model
best_category_pipeline = grid_search.best_estimator_

# Make predictions on the test set for categories
y_category_pred = best_category_pipeline.predict(X_test)

# Calculate accuracy for category prediction
category_accuracy = accuracy_score(y_category_test, y_category_pred)
print("Accuracy for category prediction: {:.2f}%".format(category_accuracy * 100))

"""# Testing"""

# New question to predict
new_question = "Where is charminar?"

# Use the trained model to predict the category
predicted_category = best_category_pipeline.predict([new_question])

print("Predicted category:", predicted_category[0])

import pickle

# Save the best performing model to a file
with open('Category_model.pkl', 'wb') as file:
    pickle.dump(best_category_pipeline, file)