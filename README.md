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


## 🚀 Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Adarshagupta/AI-Diabetes-Ditection.git
   cd AI-Diabetes-Ditection
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset:
   - Visit this link: [https://drive.google.com/file/d/1cI1d4fMrUjDFO_CRR29pXmOfNbfMSnRB/view?usp=sharing](https://drive.google.com/file/d/1cI1d4fMrUjDFO_CRR29pXmOfNbfMSnRB/view?usp=sharing)
   - Download the `diabetes_dataset00.csv` file
   - Place the downloaded file in the project root directory

5. Run the Flask application:
   ```bash
   python app.py
   ```

6. Open a web browser and navigate to `http://127.0.0.1:5000/` to use the chatbot.

## 🖥️ Usage

1. The chatbot greets you and starts asking questions about your health and lifestyle.
2. Answer each question accurately. The chatbot validates your inputs.
3. You can type 'explain' at any time to get more information about a specific factor.
4. After answering all questions, the chatbot provides a risk assessment and recommendations.
5. You can choose to check again or end the conversation.

## 🧠 Model Information

- Uses a Random Forest Classifier for risk prediction
- Features include age, BMI, blood pressure, glucose tolerance, family history, physical activity, dietary habits, and more
- Trained on the `diabetes_dataset00.csv` file (downloaded separately due to size constraints)
- Provides feature importance for top factors influencing the prediction

## 🚧 Future Improvements

- [ ] Implement user authentication for saving and retrieving past assessments
- [ ] Add more detailed visualizations of risk factors
- [ ] Integrate with wearable devices for automatic data collection
- [ ] Expand the model to cover more health conditions
- [ ] Implement a mobile app version of the chatbot
- [ ] Add multi-language support

## 👥 Contributors

- [Adarsha Gupta](https://github.com/Adarshagupta)

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This chatbot is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

## 🙏 Acknowledgements

- [Flask Documentation](https://flask.palletsprojects.com/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Diabetes Dataset Source](https://drive.google.com/file/d/1cI1d4fMrUjDFO_CRR29pXmOfNbfMSnRB/view?usp=sharing)

---

Made with ❤️ by [Adarsha Gupta](https://github.com/Adarshagupta)