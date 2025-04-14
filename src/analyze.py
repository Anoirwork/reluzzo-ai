# analyze.py
# Performs post-session analysis and identifies patterns in the data

import pandas as pd
import matplotlib.pyplot as plt

def post_session_analysis(session_data, df, model, features, success_score_minimal, actual_success_score, actual_conversion_rate, actual_sales_per_viewer, actual_engagement_per_viewer, category_mappings):
    """
    Analyze a completed session by comparing predicted and actual performance.
    
    Args:
        session_data (pd.DataFrame): DataFrame with a single row of session data.
        df (pd.DataFrame): Historical data for comparison.
        model: Trained LightGBM model.
        features (list): List of features used by the model.
        success_score_minimal (float): Minimum success score threshold.
        actual_success_score (float): Actual success score of the session.
        actual_conversion_rate (float): Actual conversion rate.
        actual_sales_per_viewer (float): Actual sales per viewer.
        actual_engagement_per_viewer (float): Actual engagement per viewer.
        category_mappings (dict): Mapping of categorical features to their categories.
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

    # Predict the success score
    predicted_success = model.predict(session_data[features])[0]

    # Get region name for reporting
    region = session_data['region'].values[0]
    region_name = {0: 'SG', 1: 'US', 2: 'KR'}.get(region, 'Other')

    # Calculate average success score for the region
    avg_success = df[df['region'] == region]['success_score'].mean()

    # Print analysis
    print(f"\nPost-Session Analysis (Region: {region_name}):")
    print(f"Actual Success Score: {actual_success_score:.2f} (out of 100)")
    print(f"Predicted Success Score: {predicted_success:.2f} (out of 100)")
    print(f"Average Success Score (for {region_name} sessions): {avg_success:.2f}")

    # Compare actual vs. predicted components
    print("\nComponent Comparison:")
    print(f"Actual Conversion Rate: {actual_conversion_rate:.2f}%")
    print(f"Actual Sales per Viewer: S${actual_sales_per_viewer:.2f}")
    print(f"Actual Engagement per Viewer: {actual_engagement_per_viewer:.2f}")

    # Provide feedback based on performance
    if actual_success_score < success_score_minimal:
        print("\nFeedback: The session underperformed. Consider the following:")
        if actual_conversion_rate < df['conversion_rate'].mean():
            print("- Conversion rate is below average. Try adding discounts or cart reservations.")
        if actual_sales_per_viewer < df['sales_per_viewer'].mean():
            print("- Sales per viewer is below average. Consider featuring higher-priced luxury items.")
        if actual_engagement_per_viewer < df['engagement_per_viewer'].mean():
            print("- Engagement per viewer is below average. Try scheduling during peak hours (e.g., 19:00-22:00).")

def identify_patterns(df):
    """
    Identify patterns in the data by analyzing feature importance and visualizing trends.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with features and success scores.
    """
    print("\nPattern Identification (Luxury Focus):")
    
    # Focus on SG data
    sg_data = df[df['region'] == 0]

    # Plot average success score by luxury tier
    plt.figure(figsize=(10, 6))
    sg_data.groupby('luxury_tier')['success_score'].mean().plot(kind='bar')
    plt.title('Average Success Score by Luxury Tier (SG)')
    plt.xlabel('Luxury Tier (1=Premium, 2=Mid-Tier, 3=Ultra-Luxury)')
    plt.ylabel('Success Score (out of 100)')
    plt.xticks(rotation=0)
    plt.show()

    # Plot average success score by start hour
    plt.figure(figsize=(10, 6))
    sg_data.groupby('start_hour')['success_score'].mean().plot(kind='bar')
    plt.title('Average Success Score by Start Hour (SG)')
    plt.xlabel('Start Hour')
    plt.ylabel('Success Score (out of 100)')
    plt.show()

    # Plot average success score by discount percentage
    plt.figure(figsize=(10, 6))
    sg_data.groupby('discount_percentage')['success_score'].mean().plot(kind='bar')
    plt.title('Average Success Score by Discount Percentage (SG)')
    plt.xlabel('Discount Percentage')
    plt.ylabel('Success Score (out of 100)')
    plt.show()