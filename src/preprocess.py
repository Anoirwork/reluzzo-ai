# preprocess.py
# Handles data loading, cleaning, and feature engineering for live session data

import pandas as pd
import numpy as np
from .config import LUXURY_TIERS, HOLIDAYS

def load_data(file_path):
    """
    Load the raw CSV data into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    # Convert timestamp columns to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    return df

def engineer_features(df):
    """
    Perform feature engineering on the raw data, adding derived features for the model.
    
    Args:
        df (pd.DataFrame): Raw DataFrame with required and optional fields.
    
    Returns:
        pd.DataFrame: DataFrame with engineered features.
    """
    # Time-based features
    # Calculate duration in minutes
    df['duration'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    # Extract hour of the day (0-23)
    df['start_hour'] = df['start_time'].dt.hour
    # Extract day of the week (0=Monday, 6=Sunday)
    df['day_of_week'] = df['start_time'].dt.dayofweek
    # Check if the session is in the evening (19:00-22:00), popular in SG
    df['is_evening_session'] = df['start_hour'].apply(lambda x: 1 if 19 <= x <= 22 else 0)

    # Language in title (simplified detection)
    # 1=English, 2=Mandarin, 0=Other
    df['language_in_title'] = df['title'].apply(
        lambda x: 1 if any(word.lower() in x.lower() for word in ['apr', 'part', 'cart'])
        else 2 if any(word in x for word in ['女孩', '水晶']) else 0
    )

    # Holiday features
    # Check if the session date matches a holiday in the region
    df['is_holiday_sg'] = df['start_time'].dt.date.astype(str).isin(HOLIDAYS['sg']).astype(int)
    df['is_holiday_us'] = df['start_time'].dt.date.astype(str).isin(HOLIDAYS['us']).astype(int)
    df['is_holiday_kr'] = df['start_time'].dt.date.astype(str).isin(HOLIDAYS['kr']).astype(int)
    # General holiday feature based on region
    df['is_holiday'] = df.apply(
        lambda x: x['is_holiday_sg'] if x['region'] == 0
        else x['is_holiday_us'] if x['region'] == 1
        else x['is_holiday_kr'] if x['region'] == 2 else 0, axis=1
    )

    # Average order value (sales_amount / number_of_orders)
    df['avg_price'] = df['sales_amount'] / df['number_of_orders'].replace(0, np.nan)

    # Luxury-specific features
    # Map brands to IDs for categorical encoding
    all_brands = set(LUXURY_TIERS['ultra_luxury'] + LUXURY_TIERS['mid_tier_luxury'] + LUXURY_TIERS['premium'])
    brand_map = {brand: idx for idx, brand in enumerate(all_brands, 1)}
    df['brand_id'] = df['title'].apply(lambda x: next((v for k, v in brand_map.items() if k in x), 0))

    # Categorize brands into luxury tiers (3=ultra-luxury, 2=mid-tier, 1=premium, 0=unknown)
    def get_luxury_tier(title):
        if any(brand in title for brand in LUXURY_TIERS['ultra_luxury']):
            return 3
        elif any(brand in title for brand in LUXURY_TIERS['mid_tier_luxury']):
            return 2
        elif any(brand in title for brand in LUXURY_TIERS['premium']):
            return 1
        return 0

    df['luxury_tier'] = df['title'].apply(get_luxury_tier)

    # Extract discount percentage from title (e.g., "10% OFF" → 10)
    df['discount_percentage'] = df['title'].apply(
        lambda x: 10 if '10% OFF' in x else 15 if '15% OFF' in x else 20 if '20% OFF' in x
        else 30 if '30% OFF' in x else 5 if '5% OFF' in x else 0
    )

    # Check if the session promotes exclusivity (e.g., "LIMITED", "EXCLUSIVE")
    df['is_exclusive'] = df['title'].str.contains('LIMITED|EXCLUSIVE|SOLD OUT', case=False).astype(int)

    # Categorize average price into bins (1=low: <S$200, 2=medium: S$200-S$1,000, 3=high: >S$1,000)
    df['avg_price_category'] = pd.cut(
        df['avg_price'], bins=[0, 200, 1000, float('inf')], labels=[1, 2, 3], include_lowest=True
    )

    # Check if the session has a cart reservation (e.g., "[CART RESERVATION")
    df['has_cart'] = df['title'].apply(lambda x: 1 if '[CART RESERVATION' in x else 0)

    # Check if the session is part of a series (e.g., "PART" in title)
    df['is_part'] = df['title'].str.contains('PART').astype(int)

    # Placeholder for number of products (update with actual data if available)
    if 'num_products' not in df.columns:
        df['num_products'] = 5  # Placeholder

    # Check if the session has a discount
    df['has_discount'] = (df['discount_percentage'] > 0).astype(int)

    # Placeholder for product category ID (update with actual data if available)
    if 'product_category_id' not in df.columns:
        df['product_category_id'] = 0  # Placeholder

    # Calculate streamer experience (number of previous sessions per streamer)
    df['streamer_experience'] = df.groupby('streamer_id').cumcount()

    # Calculate comments per minute if available
    if 'comments_per_minute' not in df.columns:
        df['comments_per_minute'] = df['total_comments'] / df['duration'].replace(0, np.nan)

    return df

def calculate_success_score(df, weights):
    """
    Calculate the success score for each session based on conversion rate, sales per viewer,
    and engagement per viewer.
    
    Args:
        df (pd.DataFrame): DataFrame with engineered features.
        weights (dict): Weights for success score components (w1, w2, w3).
    
    Returns:
        pd.DataFrame: DataFrame with success score added.
    """
    # Calculate components of the success score
    # Conversion rate: percentage of comments that resulted in a purchase
    df['conversion_rate'] = (df['purchase_comments'] / df['total_comments'].replace(0, np.nan)) * 100
    # Sales per viewer: revenue generated per viewer
    df['sales_per_viewer'] = df['sales_amount'] / df['views'].replace(0, np.nan)
    # Engagement per viewer: total engagement (comments + reactions + shares) per viewer
    df['engagement_per_viewer'] = (df['total_comments'] + df['reactions'] + df['shares']) / df['views'].replace(0, np.nan)

    # Normalize components by their 90th percentile to ensure they’re on the same scale
    conversion_norm = df['conversion_rate'].quantile(0.9)
    sales_norm = df['sales_per_viewer'].quantile(0.9)
    engagement_norm = df['engagement_per_viewer'].quantile(0.9)
    df['conversion_rate_norm'] = df['conversion_rate'] / conversion_norm
    df['sales_per_viewer_norm'] = df['sales_per_viewer'] / sales_norm
    df['engagement_per_viewer_norm'] = df['engagement_per_viewer'] / engagement_norm

    # Calculate the weighted success score
    # Success Score = w1 * conversion_rate_norm + w2 * sales_per_viewer_norm + w3 * engagement_per_viewer_norm
    df['success_score'] = (
        weights['w1'] * df['conversion_rate_norm'] +
        weights['w2'] * df['sales_per_viewer_norm'] +
        weights['w3'] * df['engagement_per_viewer_norm']
    ) * 100

    # Drop rows with NaN success scores (e.g., due to zero views or comments)
    df = df.dropna(subset=['success_score'])
    return df

def preprocess_data(file_path, weights):
    """
    Main preprocessing function that loads data, engineers features, and calculates success scores.
    
    Args:
        file_path (str): Path to the CSV file.
        weights (dict): Weights for success score components.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for training.
    """
    # Load the raw data
    df = load_data(file_path)
    
    # Engineer features
    df = engineer_features(df)
    
    # Calculate success score
    df = calculate_success_score(df, weights)
    
    return df