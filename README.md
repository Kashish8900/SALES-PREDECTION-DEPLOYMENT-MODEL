# SALES-PREDECTION-DEPLOYMENT-MODEL
A machine learning based Flask web application for predicting future sales using historical data and product features.  This project integrates CatBoost and scikit-learn for model training, pandas and numpy for data preprocessing,  and provides a clean HTML/CSS frontend for user input and visualization of predicted sales.

This project is a **Sales Prediction Web Application** built using **Machine Learning** and **Flask**.  
The application predicts product sales based on input features provided by the user through a web interface.  
The trained ML model is deployed on the web using **Render**.

## 🚀 Features
- Predicts sales using a trained Machine Learning model
- Flask-based backend
- HTML/CSS frontend
- Model loaded from a `.pkl` file
- Deployed on Render cloud platform

## 🛠️ Technologies Used
- Python
- Flask
- HTML & CSS
- Machine Learning (Scikit-learn / CatBoost)
- Gunicorn
- Render

## 📂 Project Structure
PROJECTMACHINE/
│
├── app.py
├── model.pkl
├── requirements.txt
├── Procfile
├── README.md
│
├── templates/
│ └── index.html
│
└── static/
└── style.css
