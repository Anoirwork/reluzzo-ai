# main.py
# Main script to run the luxury live session optimizer pipeline

import pandas as pd
from src.config import SUCCESS_WEIGHTS, SUCCESS_SCORE_MINIMAL
from src.preprocess import preprocess_data
from src.train import train_model, get_features
from src.predict import load_model, provide_real_time_suggestions
from src.analyze import post_session_analysis, identify_patterns

def main():
    """
    Main function to run the entire pipeline: preprocess, train, predict, and analyze.
    """
    # Define file paths
    data_path = 'data/live_sessions.csv'  # Use live_sessions_enhanced.csv for optional fields
    model_path = 'models/live_session_model.pkl'

    # Step 1: Preprocess the data
    print("Preprocessing data...")
    df = preprocess_data(data_path, SUCCESS_WEIGHTS)
    
    # Step 2: Train the model
    print("\nTraining model...")
    features = get_features()
    train_model(df, features, model_path)
    
    # Step 3: Load the model for predictions
    print("\nLoading model...")
    model, features, category_mappings = load_model(model_path)
    
    # Step 4: Provide real-time suggestions for a new session
    # Example: Create a new session (you can replace this with actual data)
    new_session = df.iloc[[0]].copy()  # Use the first row as an example
    print("\nProviding real-time suggestions for a new session...")
    current_prediction, suggestions, general_suggestions = provide_real_time_suggestions(
        new_session, df, model, features, category_mappings
    )
    print("\nField-Specific Suggestions:")
    for key, value in suggestions.items():
        print(f"- {value}")
    print("\nGeneral Suggestions:")
    for suggestion in general_suggestions:
        print(f"- {suggestion}")
    
    # Step 5: Perform post-session analysis
    # Example: Use actual values (replace with real data)
    actual_success_score = current_prediction  # Placeholder
    actual_conversion_rate = new_session['conversion_rate'].values[0]
    actual_sales_per_viewer = new_session['sales_per_viewer'].values[0]
    actual_engagement_per_viewer = new_session['engagement_per_viewer'].values[0]
    print("\nPerforming post-session analysis...")
    post_session_analysis(
        new_session, df, model, features, SUCCESS_SCORE_MINIMAL,
        actual_success_score, actual_conversion_rate, actual_sales_per_viewer,
        actual_engagement_per_viewer, category_mappings
    )
    
    # Step 6: Identify patterns in the data
    print("\nIdentifying patterns...")
    identify_patterns(df)

if __name__ == "__main__":
    main()