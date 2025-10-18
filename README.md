**Overview**

The Car Price Prediction app is a machine learning–powered web application that predicts the price of a car based on user-provided features like make, model, year, mileage, and other specifications.

It’s built using Python, Streamlit, and scikit-learn, and deployed online for real-time predictions.
## 🔗 Live Demo
👉 https://ai-on-wheels-smart-car-price-prediction-system-mczg4xkrahgvmvq.streamlit.app/
---

## 📌 Overview

The **Car Price Prediction** app is a machine learning–powered web application that predicts the **price of a car** based on user-provided features like make, model, year, mileage, and other specifications.  

It’s built using **Python**, **Streamlit**, and **scikit-learn**, and deployed online for **real-time predictions**.  

⚠️ **Disclaimer:** This app is for **educational purposes only** and should **not** be used for actual commercial decisions without proper validation.

---

## 🚀 Features

✅ Predicts car prices using trained **Regression Models**  
📊 Trained on a real-world dataset (`car_data.csv`)  
🧪 Input normalization via **StandardScaler**  
💾 Model deployment with **Pickle (.pkl)** files  
🌐 Fully deployed and accessible on **Streamlit Cloud**  
📈 Interactive, simple, and user-friendly web UI  

---

## 🧰 Tech Stack

| Layer | Tools Used |
|:------|:------------|
| **Frontend** | Streamlit |
| **Backend** | Python, scikit-learn, pandas, NumPy |
| **Machine Learning Models** | Linear Regression, Random Forest Regressor (or others) |
| **Preprocessing** | StandardScaler, LabelEncoder |
| **Deployment** | Streamlit Cloud |

---

## 📁 Project Structure

📦 car-price-predictor/
├── 📜 app.py → Streamlit web application
├── 📓 CarPrice_Project.ipynb → Notebook for EDA + model training
├── 📊 car_data.csv → Dataset used for model training
│
├── 🔍 LinearRegression_car.pkl → Trained Linear Regression model
├── 🔍 RandomForest_car.pkl → Trained Random Forest model
├── ⚖️ scaler.pkl → Pre-fitted StandardScaler
├── 📋 col.pkl → Preserved column order for inputs
│
├── 📄 requirements.txt → Python dependencies
└── 🗂️ .ipynb_checkpoints/ → Auto-saved notebook checkpoints
