# src/api.py
# Flask API to serve the luxury live session optimizer model

from flask import Flask, request, jsonify
import pandas as pd
import pickle
from src.config import SUCCESS_WEIGHTS, SUCCESS_SCORE_MINIMAL
from src.preprocess import preprocess_data, engineer_features, calculate_success_score
from src.predict import load_model, provide_real_time_suggestions
from src.analyze import post_session_analysis, identify_patterns

app = Flask(__name__)

# Global variables to store the model, features, and historical data
model = None
features = None
category_mappings = None
df = None

def initialize():
    """
    Initialize the API by loading the model, features, category mappings, and historical data.
    """
    global model, features, category_mappings, df
    
    # Load the historical data and preprocess it
    data_path = 'data/live_sessions.csv'  # Update to live_sessions_enhanced.csv if using optional fields
    df = preprocess_data(data_path, SUCCESS_WEIGHTS)
    
    # Load the model, features, and category mappings
    model_path = 'models/live_session_model.pkl'
    model, features, category_mappings = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict the success score and provide suggestions for a new live session.
    
    Request Body (JSON):
        - session_data: Dictionary containing the live session data (required fields).
    
    Response (JSON):
        - success: Boolean indicating if the request was successful.
        - predicted_score: Predicted success score (out of 100).
        - suggestions: Dictionary of field-specific suggestions.
        - general_suggestions: List of general suggestions.
        - message: Error message if the request fails.
    """
    try:
        # Get the session data from the request
        data = request.get_json()
        if not data or 'session_data' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing or invalid session_data in request body'
            }), 400

        # Convert session data to a DataFrame
        session_data = pd.DataFrame([data['session_data']])
        
        # Preprocess the session data
        session_data['start_time'] = pd.to_datetime(session_data['start_time'])
        session_data['end_time'] = pd.to_datetime(session_data['end_time'])
        session_data = engineer_features(session_data)
        session_data = calculate_success_score(session_data, SUCCESS_WEIGHTS)

        # Get predictions and suggestions
        current_prediction, suggestions, general_suggestions = provide_real_time_suggestions(
            session_data, df, model, features, category_mappings
        )

        return jsonify({
            'success': True,
            'predicted_score': current_prediction,
            'suggestions': suggestions,
            'general_suggestions': general_suggestions
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during prediction: {str(e)}'
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Endpoint to perform post-session analysis for a completed live session.
    
    Request Body (JSON):
        - session_data: Dictionary containing the live session data (required fields).
        - actual_success_score: Actual success score of the session.
        - actual_conversion_rate: Actual conversion rate.
        - actual_sales_per_viewer: Actual sales per viewer.
        - actual_engagement_per_viewer: Actual engagement per viewer.
    
    Response (JSON):
        - success: Boolean indicating if the request was successful.
        - analysis: Dictionary containing the analysis results.
        - message: Error message if the request fails.
    """
    try:
        # Get the session data and actual metrics from the request
        data = request.get_json()
        if not data or 'session_data' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing or invalid session_data in request body'
            }), 400
        if not all(key in data for key in ['actual_success_score', 'actual_conversion_rate', 'actual_sales_per_viewer', 'actual_engagement_per_viewer']):
            return jsonify({
                'success': False,
                'message': 'Missing required actual metrics in request body'
            }), 400

        # Convert session data to a DataFrame
        session_data = pd.DataFrame([data['session_data']])
        
        # Preprocess the session data
        session_data['start_time'] = pd.to_datetime(session_data['start_time'])
        session_data['end_time'] = pd.to_datetime(session_data['end_time'])
        session_data = engineer_features(session_data)
        session_data = calculate_success_score(session_data, SUCCESS_WEIGHTS)

        # Perform post-session analysis
        actual_success_score = data['actual_success_score']
        actual_conversion_rate = data['actual_conversion_rate']
        actual_sales_per_viewer = data['actual_sales_per_viewer']
        actual_engagement_per_viewer = data['actual_engagement_per_viewer']

        # Capture the analysis output (since post_session_analysis prints to console, we'll modify it to return data)
        predicted_success = model.predict(session_data[features])[0]
        region = session_data['region'].values[0]
        region_name = {0: 'SG', 1: 'US', 2: 'KR'}.get(region, 'Other')
        avg_success = df[df['region'] == region]['success_score'].mean()

        analysis = {
            'region': region_name,
            'actual_success_score': actual_success_score,
            'predicted_success_score': predicted_success,
            'average_success_score': avg_success,
            'components': {
                'actual_conversion_rate': actual_conversion_rate,
                'actual_sales_per_viewer': actual_sales_per_viewer,
                'actual_engagement_per_viewer': actual_engagement_per_viewer
            },
            'feedback': []
        }

        if actual_success_score < SUCCESS_SCORE_MINIMAL:
            if actual_conversion_rate < df['conversion_rate'].mean():
                analysis['feedback'].append("Conversion rate is below average. Try adding discounts or cart reservations.")
            if actual_sales_per_viewer < df['sales_per_viewer'].mean():
                analysis['feedback'].append("Sales per viewer is below average. Consider featuring higher-priced luxury items.")
            if actual_engagement_per_viewer < df['engagement_per_viewer'].mean():
                analysis['feedback'].append("Engagement per viewer is below average. Try scheduling during peak hours (e.g., 19:00-22:00).")

        return jsonify({
            'success': True,
            'analysis': analysis
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error during analysis: {str(e)}'
        }), 500

@app.route('/patterns', methods=['GET'])
def patterns():
    """
    Endpoint to identify patterns in the historical data.
    
    Response (JSON):
        - success: Boolean indicating if the request was successful.
        - patterns: Dictionary containing identified patterns.
        - message: Error message if the request fails.
    """
    try:
        # Identify patterns in the data
        sg_data = df[df['region'] == 0]  # Focus on SG data
        patterns = {
            'success_by_luxury_tier': sg_data.groupby('luxury_tier')['success_score'].mean().to_dict(),
            'success_by_start_hour': sg_data.groupby('start_hour')['success_score'].mean().to_dict(),
            'success_by_discount_percentage': sg_data.groupby('discount_percentage')['success_score'].mean().to_dict()
        }

        return jsonify({
            'success': True,
            'patterns': patterns
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error identifying patterns: {str(e)}'
        }), 500

if __name__ == "__main__":
    # Initialize the API
    initialize()
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)