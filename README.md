# **Machine Learning Project** 
# 💎 Diamond Price Prediction

### Live deployed Link: https://gemstone-price-prediction-isec.onrender.com/

## 📌 Overview
This project focuses on predicting the **price of diamonds** using **machine learning regression models** based on their physical and quality-related attributes.  
It demonstrates an **end-to-end ML workflow**, from data preprocessing and model training to deployment using Flask.

---

## 🎯 Problem Statement
Diamond prices depend on multiple factors such as carat, cut, color, and clarity.  
Manual estimation is often inaccurate and inconsistent.

**Goal:**  
Build a machine learning model that accurately predicts diamond prices based on given features and provides predictions through a web interface.

---

## 📊 Dataset Description
The dataset contains information about diamonds and their corresponding market prices.

### 🔹 Features
- **Carat** – Weight of the diamond  
- **Cut** – Quality of the cut (Fair, Good, Very Good, Premium, Ideal)  
- **Color** – Diamond color grading (D to J)  
- **Clarity** – Purity level (IF, VVS1, VVS2, VS1, VS2, SI1, SI2, I1)  
- **Depth** – Total depth percentage  
- **Table** – Width of the top of the diamond  
- **X** – Length (mm)  
- **Y** – Width (mm)  
- **Z** – Height (mm)  

### 🎯 Target Variable
- **Price** – Market price of the diamond (continuous value)

---

## 🧠 Machine Learning Approach
- **Problem Type:** Regression  
- **Steps involved:**
  - Data preprocessing and feature engineering
  - Encoding categorical features
  - Feature scaling
  - Training multiple regression models
  - Selecting the best-performing model

---

## 📈 Model Evaluation
The model is evaluated using standard regression metrics:
- **R² Score**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

---

## 🔬 Experiment Tracking
- **MLflow** is used to:
  - Log model parameters
  - Track evaluation metrics
  - Store trained models
- Experiments are tracked via **DAGsHub MLflow UI**

---

## 🌐 Web Application
A **Flask-based web app** allows users to:
- Enter diamond attributes
- Get real-time price predictions
- View results in a clean and responsive UI

---

## 🚀 Deployment
- Deployed on **Render**
- Uses **Gunicorn** for production
- Linux-compatible dependencies

---

## 🛠 Tech Stack
- **Language:** Python  
- **Libraries:** NumPy, Pandas, Scikit-learn  
- **Experiment Tracking:** MLflow, DAGsHub  
- **Web Framework:** Flask  
- **Deployment:** Render  
- **Version Control:** Git, GitHub  

---

## 📂 Project Structure
Diamond-Price-Prediction/
│
├── src/
│ ├── components/
│ ├── pipelines/
│ └── utils/
│
├── templates/
│ ├── index.html
│ └── home.html
│
├── app.py
├── requirements.txt
├── README.md
└── artifacts/


---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/Diamond-Price-Prediction.git
cd Diamond-Price-Prediction

python -m venv venv
venv\Scripts\activate   # Windows

pip install -r requirements.txt

python app.py

run above lines line by line in cmd


