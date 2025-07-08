# RecommendCare

## Overview
RecommendCare is a web application designed to provide personalized health recommendations based on user input. It leverages machine learning models and natural language processing to generate tailored advice for various health conditions.

## Features
- **User Input Prediction**: Predicts health metrics like systolic and diastolic blood pressure and glucose levels based on user data.
- **Personalized Recommendations**: Provides health advice based on predicted metrics and user conditions.
- **Interactive Web Interface**: Allows users to input their data and view recommendations.

## Project Structure
- `app.py`: Flask application that handles user input, prediction, and recommendation generation.
- `userInputPred.py`: Contains the machine learning model for predicting health metrics.
- `ragSource.txt`: A text file containing health recommendations for various conditions.
- `templates/index.html`: HTML template for the web interface.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Navigate to the project directory:
   ```bash
   cd CM Project
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`.
3. Input your data and view personalized health recommendations.

## Acknowledgments
- Kaggle for the dataset.
- LangChain for NLP tools.
- Scikit-learn for machine learning utilities.
