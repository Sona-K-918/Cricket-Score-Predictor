import pandas as pd
import pickle

def test_predict():
    print("Loading cricket_score_predictor.pkl...")
    try:
        with open('cricket_score_predictor.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test data
    sample = pd.DataFrame({
        'bat_team': ['Royal Challengers Bangalore'],
        'bowl_team': ['Kolkata Knight Riders'],
        'venue': ['M Chinnaswamy Stadium'],
        'runs': [100],
        'wickets': [3],
        'overs': [10.0],
        'runs_last_5': [45],
        'wickets_last_5': [1]
    })
    
    print("\nRunning sample prediction...")
    print(f"Inputs:\n{sample.to_string(index=False)}")
    
    try:
        prediction = model.predict(sample)
        predicted_score = round(prediction[0])
        print(f"\nPrediction Success! Predicted Final Score: {predicted_score}")
        print(f"Predicted Range: {predicted_score - 5} - {predicted_score + 5}")
    except Exception as e:
        print(f"Error during prediction: {e}")

if __name__ == '__main__':
    test_predict()
