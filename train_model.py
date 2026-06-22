import json
import pickle
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train():
    print("Loading Dataset/ipl.csv...")
    df = pd.read_csv('Dataset/ipl.csv')
    
    # Cleaning as in notebook
    df = df[df['overs'] <= 20]
    df = df[df['wickets'] <= 10]
    
    # Feature selection
    X = df[[
        'bat_team',
        'bowl_team',
        'venue',
        'runs',
        'wickets',
        'overs',
        'runs_last_5',
        'wickets_last_5'
    ]]
    y = df['total']
    
    # Metadata extraction for Streamlit
    meta_info = {
        'teams': sorted(list(df['bat_team'].unique())),
        'venues': sorted(list(df['venue'].unique())),
        'min_runs': int(df['runs'].min()),
        'max_runs': int(df['runs'].max()),
        'min_wickets': int(df['wickets'].min()),
        'max_wickets': int(df['wickets'].max()),
        'min_overs': float(df['overs'].min()),
        'max_overs': float(df['overs'].max()),
        'min_runs_last_5': int(df['runs_last_5'].min()),
        'max_runs_last_5': int(df['runs_last_5'].max()),
        'min_wickets_last_5': int(df['wickets_last_5'].min()),
        'max_wickets_last_5': int(df['wickets_last_5'].max()),
    }
    
    # Save meta info
    with open('meta_info.json', 'w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=4)
    print("Saved meta_info.json.")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Preprocessor
    categorical_cols = ['venue', 'bat_team', 'bowl_team']
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    # Model pipeline
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    
    print("Fitting model pipeline...")
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Evaluation Metrics on Test Set:")
    print(f"  MAE : {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R2  : {r2:.4f}")
    
    # Save model
    with open('cricket_score_predictor.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Saved cricket_score_predictor.pkl successfully!")

if __name__ == '__main__':
    train()
