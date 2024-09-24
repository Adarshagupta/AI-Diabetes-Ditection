import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import uuid
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

logging.basicConfig(level=logging.INFO)

# Load the dataset
data = pd.read_csv('diabetes_dataset00.csv')
logging.info(f"Dataset columns: {data.columns}")


# Assuming the first column is the target variable
y = data.iloc[:, 0]
X = data.iloc[:, 1:]

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create a pipeline with Random Forest Classifier
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit the model
model.fit(X, y)

# Save the model
joblib.dump(model, 'diabetes_model.joblib')

# Define prompts and expected answer types for each feature
prompts = {
    'Age': {"question": "What is your age?", "type": "int", "min": 0, "max": 120},
    'BMI': {"question": "What is your BMI (Body Mass Index)?", "type": "float", "min": 10, "max": 50},
    'Blood Pressure': {"question": "What is your blood pressure (systolic/diastolic)?", "type": "str", "format": r'\d{2,3}/\d{2,3}'},
    'Glucose Tolerance Test': {"question": "What was your glucose tolerance test result (in mg/dL)?", "type": "float", "min": 50, "max": 300},
    'Family History': {"question": "Do you have a family history of diabetes? (Yes/No)", "type": "bool"},
    'Physical Activity': {"question": "How many minutes of physical activity do you do per week?", "type": "int", "min": 0, "max": 1000},
    'Dietary Habits': {"question": "On a scale of 1-10, how healthy would you rate your diet?", "type": "int", "min": 1, "max": 10},
    'Smoking Status': {"question": "Are you a smoker? (Yes/No)", "type": "bool"},
    'Alcohol Consumption': {"question": "How many alcoholic drinks do you consume per week?", "type": "int", "min": 0, "max": 100},
    'Waist Circumference': {"question": "What is your waist circumference in inches?", "type": "float", "min": 20, "max": 80},
    'Sleep Duration': {"question": "How many hours of sleep do you get on average per night?", "type": "float", "min": 0, "max": 24},
    'Stress Level': {"question": "On a scale of 1-10, how would you rate your stress level?", "type": "int", "min": 1, "max": 10},
    'Genetic Risk': {"question": "Have you been told you have genetic risk factors for diabetes? (Yes/No)", "type": "bool"}
}


@app.route('/')
def home():
    session['user_id'] = str(uuid.uuid4())
    session['current_feature'] = 0
    session['input_data'] = {}
    return render_template('index.html', features=list(prompts.keys()), prompts=prompts)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame([input_data])
    
    # Ensure input_df has the same columns as the training data
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = np.nan
    
    input_df = input_df[X.columns]
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    feature_importances = dict(zip(X.columns, model.named_steps['classifier'].feature_importances_))
    top_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
    
    result = "high" if prediction == 1 else "low"
    response = f"Based on the provided information, your risk of diabetes is {result}. "
    response += f"The probability of having diabetes is {probability:.2f}. "
    response += "The top 5 factors influencing this prediction are:\n"
    for feature, importance in top_features:
        response += f"- {feature}: {importance:.2f}\n"
    
    recommendations = generate_recommendations(input_data, result)
    response += "\nRecommendations:\n" + "\n".join(recommendations)
    
    # Compare with average population
    avg_risk = model.predict_proba(X).mean(axis=0)[1]
    response += f"\n\nYour risk compared to the average population: "
    response += f"{'Higher' if probability > avg_risk else 'Lower'}"
    response += f" (Your risk: {probability:.2f}, Average risk: {avg_risk:.2f})"
    
    return jsonify({"response": response})

@app.route('/explain', methods=['POST'])
def explain_factor():
    factor = request.json['factor']
    explanation = get_factor_explanation(factor)
    return jsonify({"explanation": explanation})

def get_factor_explanation(factor):
    explanations = {
        'Age': "Age is a significant factor in diabetes risk. As you get older, the risk of developing type 2 diabetes increases.",
        'BMI': "BMI (Body Mass Index) is a measure of body fat based on height and weight. A higher BMI is associated with increased diabetes risk.",
        'Blood Pressure': "High blood pressure is often associated with insulin resistance, which can lead to diabetes.",
        'Glucose Tolerance Test': "This test measures how well your body processes glucose. Higher results indicate a greater risk of diabetes.",
        'Family History': "Having a close family member with diabetes increases your risk of developing the condition.",
        'Physical Activity': "Regular physical activity helps maintain a healthy weight and improves insulin sensitivity, reducing diabetes risk.",
        'Dietary Habits': "A diet high in processed foods and sugar can increase diabetes risk, while a balanced diet can help prevent it.",
        'Smoking Status': "Smoking can increase insulin resistance and the risk of developing type 2 diabetes.",
        'Alcohol Consumption': "Excessive alcohol consumption can increase diabetes risk and interfere with blood sugar control.",
        'Waist Circumference': "A larger waist circumference is associated with increased risk of insulin resistance and diabetes.",
        'Sleep Duration': "Both too little and too much sleep can affect insulin sensitivity and increase diabetes risk.",
        'Stress Level': "Chronic stress can affect hormone levels and lead to insulin resistance, increasing diabetes risk.",
        'Genetic Risk': "Certain genetic factors can increase your susceptibility to developing diabetes."
    }
    return explanations.get(factor, "No explanation available for this factor.")

def generate_recommendations(input_data, risk_level):
    recommendations = []
    if risk_level == "high":
        if float(input_data.get('BMI', 0)) > 25:
            recommendations.append("Consider losing weight to reach a healthy BMI.")
        if int(input_data.get('Physical Activity', 0)) < 150:
            recommendations.append("Aim for at least 150 minutes of moderate physical activity per week.")
        if int(input_data.get('Dietary Habits', 0)) < 7:
            recommendations.append("Improve your diet by including more fruits, vegetables, and whole grains.")
        if input_data.get('Smoking Status', '').lower() == 'yes':
            recommendations.append("Consider quitting smoking to reduce your diabetes risk.")
        if float(input_data.get('Sleep Duration', 0)) < 7 or float(input_data.get('Sleep Duration', 0)) > 9:
            recommendations.append("Aim for 7-9 hours of sleep per night for optimal health.")
    else:
        recommendations.append("Maintain your current healthy lifestyle.")
        recommendations.append("Continue regular check-ups with your healthcare provider.")
    
    recommendations.append("Stay hydrated and manage stress through relaxation techniques.")
    return recommendations

if __name__ == '__main__':
    app.run(debug=True)
