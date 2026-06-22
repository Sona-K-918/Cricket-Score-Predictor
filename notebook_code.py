# -- Data Handling --
import pandas as pd
import numpy as np

# -- Visualization --
import matplotlib.pyplot as plt
import seaborn as sns

# -- Model Development --
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -- Model Evaluation --
from sklearn.model_selection import train_test_split

# --- CELL ---

from google.colab import files

uploaded = files.upload()

# --- CELL ---

import os

print(os.listdir('/content'))

# --- CELL ---

from google.colab import drive
drive.mount('/content/drive')

# --- CELL ---

df = pd.read_csv('ipl.csv')

print("First 5 rows of the DataFrame:")
df.head()

# --- CELL ---

print("\nLast 5 rows of the DataFrame:")
df.tail()

# --- CELL ---

# -- Shape --
print(df.shape)

# --- CELL ---

# -- Columns & their Data Types--
print("Column Names & Data Types :")
print(df.dtypes)

# --- CELL ---

# -- Info --
print(df.info())

# --- CELL ---

# -- Missing values --
print("Missing Values Record: ")
print(df.isnull().sum())

# --- CELL ---

# -- Statistics --
print("\nSummary statistics of the DataFrame:")
print(df.describe())

# --- CELL ---

print(f"Batting Team:\n{df['bat_team'].unique()}\n")
print(f"Bowling Team:\n{df['bowl_team'].unique()}")

# --- CELL ---

print(df['venue'].value_counts())

# --- CELL ---

plt.figure(figsize=(8,5))
sns.histplot(df['total'], kde=True)
plt.title("Distribution of Final Scores")
plt.show()

# --- CELL ---

numeric_df = df.select_dtypes(include=np.number)

plt.figure(figsize=(10,6))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title("Feature Correlation")
plt.show()

# --- CELL ---

plt.figure(figsize=(8,5))
sns.scatterplot(
    x='runs',
    y='total',
    data=df
)
plt.title("Runs vs Final Score")
plt.show()

# --- CELL ---

plt.figure(figsize=(8,5))
sns.scatterplot(
    x='wickets',
    y='total',
    data=df
)
plt.title("Wickets vs Final Score")
plt.show()

# --- CELL ---

plt.figure(figsize=(10,5))
df['venue'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Venues")
plt.show()

# --- CELL ---

plt.figure(figsize=(10,5))
df['bat_team'].value_counts().plot(kind='bar')
plt.title("Batting Teams")
plt.show()

# --- CELL ---

df.isnull().sum()

# --- CELL ---

df = df[df['overs'] <= 20]

# --- CELL ---

df = df[df['wickets'] <= 10]

# --- CELL ---

X = df[
    [
        'venue',
        'bat_team',
        'bowl_team',
        'runs',
        'wickets',
        'overs',
        'runs_last_5',
        'wickets_last_5'
    ]
]

y = df['total']

# --- CELL ---

categorical_cols = [
    'venue',
    'bat_team',
    'bowl_team'
]

# --- CELL ---

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_cols
        )
    ],
    remainder='passthrough'
)

# --- CELL ---

plt.figure(figsize=(8,5))
sns.boxplot(x=df['total'])
plt.title("Outlier Detection")
plt.show()

# --- CELL ---

X = df[
    [
        'bat_team',
        'bowl_team',
        'venue',
        'runs',
        'wickets',
        'overs',
        'runs_last_5',
        'wickets_last_5'
    ]
]

# --- CELL ---

y = df['total']

# --- CELL ---

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --- CELL ---

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# --- CELL ---

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

categorical_cols = [
    'venue',
    'bat_team',
    'bowl_team'
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_cols
        )
    ],
    remainder='passthrough'
)

# --- CELL ---

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# --- CELL ---

model.fit(X_train, y_train)

# --- CELL ---

y_pred = model.predict(X_test)

# --- CELL ---

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

# --- CELL ---

mae = mean_absolute_error(y_test, y_pred)
print("MAE:", mae)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

# --- CELL ---

sample = pd.DataFrame({
    'venue': ['Feroz Shah Kotla'],
    'bat_team': ['Royal Challengers Bangalore'],
    'bowl_team': ['Kolkata Knight Riders'],
    'runs': [100],
    'wickets': [0],
    'overs': [6],
    'runs_last_5': [88],
    'wickets_last_5': [0],
    'striker': [80],
    'non-striker': [20]
})

prediction = model.predict(sample)

print("Predicted Final Score:", round(prediction[0]))

# --- CELL ---

import pickle

with open('cricket_score_predictor.pkl', 'wb') as file:
    pickle.dump(model, file)