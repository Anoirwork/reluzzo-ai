# train.py
# Trains the LightGBM model on preprocessed data and saves it

import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

def train_model(df, features, model_path):
    """
    Train a LightGBM model on the preprocessed data and save it.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with features and success score.
        features (list): List of feature names to use for training.
        model_path (str): Path to save the trained model.
    """
    # Define the features and target
    X = df[features]
    y = df['success_score']

    # Split the data into training and testing sets (90% train, 10% test for small dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define categorical features for LightGBM
    categorical_features = [
        'region', 'brand_id', 'has_cart', 'is_part', 'num_products', 'has_discount',
        'product_category_id', 'streamer_id', 'language_in_title', 'is_holiday',
        'luxury_tier', 'is_exclusive', 'avg_price_category'
    ]

    # Convert categorical features to the correct type and store categories
    category_mappings = {}
    for col in categorical_features:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
        # Store the categories seen during training
        category_mappings[col] = X_train[col].cat.categories

    # Initialize and train the LightGBM model with adjusted parameters
    model = lgb.LGBMRegressor(
        n_estimators=50,  # Reduce number of trees for small dataset
        max_depth=3,      # Limit tree depth to prevent overfitting
        min_data_in_leaf=1,  # Allow splits with small leaves (since dataset is small)
        random_state=42
    )
    model.fit(X_train, y_train, categorical_feature=categorical_features)

    # Evaluate the model on the test set
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"Test Predictions (first 5): {y_pred[:5]}")
    print(f"Test Actual (first 5): {y_test.values[:5]}")

    # Save the model, features, and category mappings
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'features': features, 'category_mappings': category_mappings}, f)
    print(f"Model saved to {model_path}")

def get_features():
    """
    Define the list of features to use for training the model.
    
    Returns:
        list: List of feature names.
    """
    features = [
        'duration', 'start_hour', 'day_of_week', 'region', 'brand_id', 'has_cart',
        'is_part', 'avg_price', 'num_products', 'has_discount', 'product_category_id',
        'streamer_id', 'is_evening_session', 'language_in_title', 'is_holiday',
        'luxury_tier', 'discount_percentage', 'is_exclusive', 'avg_price_category',
        'streamer_experience'
    ]
    return features