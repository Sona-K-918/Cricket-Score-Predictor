# 🏏 IPL Cricket Score Predictor


## 📌 Overview

### This project predicts the final score of an IPL batting innings based on the current match situation using Machine Learning.

### The model is trained on historical IPL match data and uses features such as venue, batting team, bowling team, current score, wickets fallen, overs completed, and recent performance in the last 5 overs.


## 🎯 Objective

### To estimate the final innings total of a team during an IPL match using a Linear Regression model.


## 📊 Dataset
Historical IPL match dataset
Total Records: 76,014
Features Used:
Venue
Batting Team
Bowling Team
Current Runs
Wickets
Overs
Runs in Last 5 Overs
Wickets in Last 5 Overs

## 🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-Learn
Jupyter Notebook / Google Colab

## 🤖 Machine Learning Model

Linear Regression

Data Preprocessing
Removed irrelevant columns
One-Hot Encoding for categorical features
Train-Test Split
Model Training and Evaluation

## 📈 Model Performance
Metric	Value
MAE (Mean Absolute Error)	14.41
R² Score	0.56
Interpretation
On average, predictions are off by approximately 14 runs.
The model explains about 56% of the variance in final scores.


## 🔮 Sample Prediction
Input
{
    'venue': 'M Chinnaswamy Stadium',
    'bat_team': 'Royal Challengers Bangalore',
    'bowl_team': 'Mumbai Indians',
    'runs': 120,
    'wickets': 3,
    'overs': 15.2,
    'runs_last_5': 45,
    'wickets_last_5': 1
}

Output
Predicted Final Score: 194


## 🚀 How to Run
Clone the repository
git clone https://github.com/your-username/cricket-score-predictor.git

### Install dependencies
pip install pandas numpy scikit-learn
Open the notebook
jupyter notebook
Run all cells.


## ⚠️ Limitations
Dataset contains IPL matches only up to 2021.
Weather conditions are not considered.
Pitch reports are not included.
Player form and playing XI are not considered.
Newer teams such as Gujarat Titans (GT) and Lucknow Super Giants (LSG) are not present in the dataset.


## 📚 Future Improvements
Add Weather API integration
Include player statistics
Add win probability prediction
Build an interactive Streamlit dashboard
Use advanced models such as Random Forest and XGBoost


## 👩‍💻 Author

Sonakshi Kaushik

B.Tech CSE | Machine Learning Enthusiast | Cricket Analytics Explorer
