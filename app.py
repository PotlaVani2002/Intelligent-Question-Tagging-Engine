from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

app = Flask(__name__)

# Load the vectorizer
"""with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)"""

# Load the model
with open('taxonomy_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('sub category_model.pkl', 'rb') as file:
    model2=pickle.load(file)

with open('Difficulty_model.pkl', 'rb') as file:
    model3=pickle.load(file)

with open('Category_model.pkl', 'rb') as file:
    model4=pickle.load(file)


# Define the home route
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/index')
def index():
    return render_template('index.html')
# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the input question from the form
    question = request.form['question']
    if question.strip() == '':
        return render_template('index.html', message='Please enter question.')
    predicted_taxonomy = model.predict([question])
    predicted_category=model4.predict([question])
    predicted_difficulty=model3.predict([question])
    predicted_subCategory=model2.predict([question])

    
    # Vectorize the input question
    #question_vector = vectorizer.transform([question])
    
    # Make predictions using the model
    #predicted_taxonomy = model.predict(question_vector)
    list=[predicted_taxonomy[0],predicted_difficulty[0],predicted_category[0],predicted_subCategory[0]]

    # Render the prediction result
    #return render_template('result.html',question=question,Predicted=list)
    return render_template('result.html',Question=question,Predicted_Taxonomy=list[0],Predicted_category=list[3],Predicted_Difficulty=list[1],Predicted_subCategory=list[3])

if __name__ == '__main__':
    app.run(debug=True)
