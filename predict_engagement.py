import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('engagement_model.pkl')
scaler = joblib.load('scaler.pkl')

# Total number of features the model expects (after encoding)
expected_number_of_features = 15

# Dictionary to convert prediction back to label
labels = {0: "Low", 1: "Medium", 2: "High"}

def get_user_input():
    print("Please enter the following player details:")
    
    # Numerical features (used as-is)
    age = int(input("Age: "))
    play_time = float(input("PlayTimeHours: "))
    purchases = int(input("InGamePurchases: "))
    sessions = int(input("SessionsPerWeek: "))
    duration = int(input("AvgSessionDurationMinutes: "))

    # Categorical features (need one-hot encoding)
    location = input("Location (USA, Europe, Other): ").strip().lower()
    gender = input("Gender (Male, Female, Other): ").strip().lower()
    genre = input("GameGenre (RPG, Action, Strategy, Sports): ").strip().lower()
    difficulty = input("GameDifficulty (Easy, Medium, Hard): ").strip().lower()

    # ----------------------------
    # One-Hot Encode Categorical Inputs (must match training!)
    # Note: `drop_first=True` in training means one category per feature was dropped as baseline
    # ----------------------------

    # Location → baseline = "Other"
    location_europe = 1 if location == "europe" else 0
    location_usa = 1 if location == "usa" else 0

    # Gender → baseline = "Female"
    gender_male = 1 if gender == "male" else 0
    gender_other = 1 if gender == "other" else 0

    # GameGenre → baseline = "Action"
    genre_rpg = 1 if genre == "rpg" else 0
    genre_simulation = 1 if genre == "simulation" else 0
    genre_sports = 1 if genre == "sports" else 0
    genre_strategy = 1 if genre == "strategy" else 0

    # GameDifficulty → baseline = "Easy"
    difficulty_hard = 1 if difficulty == "hard" else 0
    difficulty_medium = 1 if difficulty == "medium" else 0

    # ----------------------------
    # Final Input Order (MUST match training order)
    # ----------------------------
    X = [
        age,                    # 0
        play_time,              # 1
        purchases,              # 2
        sessions,               # 3
        duration,               # 4
        location_europe,        # 5
        location_usa,           # 6
        gender_male,            # 7
        gender_other,           # 8
        genre_rpg,              # 9
        genre_simulation,       # 10
        genre_sports,           # 11
        genre_strategy,         # 12
        difficulty_hard,        # 13
        difficulty_medium       # 14
    ]

    # Final sanity check (optional)
    if len(X) != expected_number_of_features:
        raise ValueError(f"Feature count mismatch! Got {len(X)} features, expected {expected_number_of_features}.")

    return np.array(X).reshape(1, -1)

# Main program
if __name__ == "__main__":
    X_input = get_user_input()
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)

    print(f"\n Predicted Engagement Level: {labels[prediction[0]]}")
