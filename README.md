# Advanced Diabetes Risk Chatbot

## Project Overview
This project implements an advanced chatbot that assesses a user's risk of diabetes based on various health and lifestyle factors. The chatbot uses a machine learning model to predict the risk and provides personalized recommendations.

## Features
- Interactive chat interface for data collection
- Machine learning-based risk assessment
- Personalized recommendations based on risk level
- Input validation to ensure data quality
- Detailed explanation of risk factors


## Technologies Used
- Python 3.8+
- Flask 2.1.0
- pandas 1.3.5

- numpy 1.21.5
- scikit-learn 1.0.2
- joblib 1.1.0
- HTML/CSS
- JavaScript (jQuery)

## Project Structure
- `app.py`: Main Flask application and machine learning model
- `templates/index.html`: Frontend chat interface
- `requirements.txt`: List of Python dependencies
- `diabetes_dataset00.csv`: Dataset for training the model (not included in repository)
- `diabetes_model.joblib`: Saved machine learning model (generated after running the app)

## Setup and Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd advanced-diabetes-risk-chatbot
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Ensure you have the `diabetes_dataset00.csv` file in the project root directory.

5. Run the Flask application:
   ```
   python app.py
   ```

6. Open a web browser and navigate to `http://127.0.0.1:5000/` to use the chatbot.

## Usage
1. The chatbot will greet you and start asking questions about your health and lifestyle.
2. Answer each question accurately. The chatbot will validate your inputs.
3. After answering all questions, the chatbot will provide a risk assessment and recommendations.
4. You can choose to check again or end the conversation.

## Model Information
- The project uses a Random Forest Classifier for risk prediction.
- Features include age, BMI, blood pressure, glucose tolerance, family history, physical activity, dietary habits, and more.
- The model is trained on the `diabetes_dataset00.csv` file (not included in the repository due to data privacy).

## Future Improvements
- Implement user authentication for saving and retrieving past assessments
- Add more detailed explanations for each risk factor
- Integrate with wearable devices for automatic data collection
- Expand the model to cover more health conditions

## Contributors
- [Your Name]

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This chatbot is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
