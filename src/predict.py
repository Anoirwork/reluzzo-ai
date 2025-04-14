# predict.py
# Provides real-time predictions and suggestions for new live sessions

import pandas as pd
import pickle
from .config import SUCCESS_SCORE_MINIMAL

def load_model(model_path):
    """
    Load the trained model, features, and category mappings from a pickle file.
    
    Args:
        model_path (str): Path to the saved model.
    
    Returns:
        tuple: (model, features, category_mappings) - The loaded model, features, and category mappings.
    """
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
    return saved_data['model'], saved_data['features'], saved_data.get('category_mappings', {})

def provide_real_time_suggestions(session_data, df, model, features, category_mappings):
    """
    Provide real-time predictions and suggestions for a new live session.
    
    Args:
        session_data (pd.DataFrame): DataFrame with a single row of session data.
        df (pd.DataFrame): Historical data for calculating quantiles and suggestions.
        model: Trained LightGBM model.
        features (list): List of features used by the model.
        category_mappings (dict): Mapping of categorical features to their categories.
    
    Returns:
        tuple: (current_prediction, suggestions, general_suggestions)
    """
    # Define categorical features
    categorical_features = [
        'region', 'brand_id', 'has_cart', 'is_part', 'num_products', 'has_discount',
        'product_category_id', 'streamer_id', 'language_in_title', 'is_holiday',
        'luxury_tier', 'is_exclusive', 'avg_price_category'
    ]

    # Ensure session_data has the correct categorical features
    for col in categorical_features:
        if col in category_mappings:
            session_data[col] = pd.Categorical(session_data[col], categories=category_mappings[col])

    # Predict the success score for the current session
    current_prediction = model.predict(session_data[features])[0]
    print(f"\nPredicted Success Score: {current_prediction:.2f} (out of 100)")

    # Initialize suggestions
    suggestions = {}
    general_suggestions = []

    # Calculate quantiles for numerical features to set reasonable bounds
    duration_min, duration_max = df['duration'].quantile([0.1, 0.9])
    avg_price_min, avg_price_max = df['avg_price'].quantile([0.1, 0.9])

    # Iterate over each feature to test alternative values and suggest improvements
    for feature in session_data.columns:
        value = session_data[feature].values[0]
        best_value = value
        best_prediction = current_prediction

        # Test numerical features by adjusting their values
        if feature in ['duration', 'avg_price', 'discount_percentage']:
            min_val = duration_min if feature == 'duration' else avg_price_min if feature == 'avg_price' else 0
            max_val = duration_max if feature == 'duration' else avg_price_max if feature == 'avg_price' else 30
            test_values = [value * 0.8, value * 1.2, value + 10, value + 30] if feature != 'discount_percentage' else [0, 5, 10, 15, 20, 30]
            test_values = [max(min_val, min(max_val, v)) for v in test_values if v >= 0]

            for test_value in test_values:
                temp_data = session_data.copy()
                temp_data[feature] = test_value
                # Reapply categorical feature types
                for col in categorical_features:
                    if col in category_mappings:
                        temp_data[col] = pd.Categorical(temp_data[col], categories=category_mappings[col])
                new_prediction = model.predict(temp_data[features])[0]
                if new_prediction > best_prediction:
                    best_prediction = new_prediction
                    best_value = test_value
            if best_value != value:
                suggestions[feature] = f"Adjust {feature} to {best_value:.2f} (Predicted success: {best_prediction:.2f})"

        # Test time-based features (start_hour, day_of_week)
        elif feature in ['start_hour', 'day_of_week']:
            test_range = range(0, 24) if feature == 'start_hour' else range(0, 7)
            for test_value in test_range:
                if test_value == value:
                    continue
                temp_data = session_data.copy()
                temp_data[feature] = test_value
                for col in categorical_features:
                    if col in category_mappings:
                        temp_data[col] = pd.Categorical(temp_data[col], categories=category_mappings[col])
                new_prediction = model.predict(temp_data[features])[0]
                if new_prediction > best_prediction:
                    best_prediction = new_prediction
                    best_value = test_value
            if best_value != value:
                suggestions[feature] = f"Change {feature} to {best_value} (Predicted success: {best_prediction:.2f})"

        # Test binary/categorical features
        elif feature in ['region', 'has_cart', 'is_part', 'has_discount', 'is_evening_session', 'is_holiday', 'is_exclusive', 'brand_id', 'product_category_id', 'streamer_id', 'language_in_title', 'luxury_tier', 'avg_price_category']:
            test_values = df[feature].unique() if feature in categorical_features else [1 if value == 0 else 0]
            for test_value in test_values:
                if test_value == value:
                    continue
                temp_data = session_data.copy()
                temp_data[feature] = test_value
                for col in categorical_features:
                    if col in category_mappings:
                        temp_data[col] = pd.Categorical(temp_data[col], categories=category_mappings[col])
                new_prediction = model.predict(temp_data[features])[0]
                if new_prediction > best_prediction:
                    best_prediction = new_prediction
                    best_value = test_value
            if best_value != value:
                if feature in ['region', 'brand_id', 'product_category_id', 'streamer_id', 'language_in_title', 'luxury_tier', 'avg_price_category']:
                    suggestions[feature] = f"Switch to {feature} {best_value} (Predicted success: {best_prediction:.2f})"
                else:
                    action = "Add" if best_value == 1 else "Remove"
                    suggestions[feature] = f"{action} {feature} (Predicted success: {best_prediction:.2f})"

    # General suggestions if the success score is below the threshold
    if current_prediction < SUCCESS_SCORE_MINIMAL:
        general_suggestions.append(f"Overall success score is below target ({SUCCESS_SCORE_MINIMAL}). Consider optimizing key factors.")
    if 'has_cart' not in suggestions and session_data['has_cart'].values[0] == 0:
        general_suggestions.append("Consider adding a cart reservation to create urgency.")
    if 'is_evening_session' not in suggestions and session_data['is_evening_session'].values[0] == 0:
        general_suggestions.append("Consider scheduling the session between 19:00-22:00 for better engagement in SG.")
    if 'is_exclusive' not in suggestions and session_data['is_exclusive'].values[0] == 0:
        general_suggestions.append("Consider adding 'LIMITED' or 'EXCLUSIVE' to the title to appeal to luxury buyers.")

    return current_prediction, suggestions, general_suggestions