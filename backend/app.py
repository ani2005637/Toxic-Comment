"""
Toxic Comments Classification API
Main Flask application for serving the toxic comment classifier model
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import re
from datetime import datetime
import json
try:
    from google import genai
except ImportError:
    genai = None

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing for frontend communication

# Global variables for model and vectorizer
model = None
vectorizer = None

# Configuration
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'toxic_classifier.pkl')
VECTORIZER_PATH = os.path.join(os.path.dirname(__file__), 'models', 'vectorizer.pkl')

# Toxicity categories
TOXICITY_LABELS = [
    'toxic',
    'severe_toxic',
    'obscene',
    'threat',
    'insult',
    'identity_hate'
]


def load_models():
    """
    Load the pre-trained model and vectorizer from disk
    Creates dummy models if files don't exist (for initial setup)
    """
    global model, vectorizer
    
    try:
        # Check if model files exist
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            with open(VECTORIZER_PATH, 'rb') as f:
                vectorizer = pickle.load(f)
            print("[OK] Models loaded successfully!")
        else:
            print("[WARN] Model files not found. Using demo mode.")
            print("  Run train_model.py to train the actual model.")
            model = None
            vectorizer = None
    except Exception as e:
        print(f"[ERROR] Error loading models: {str(e)}")
        model = None
        vectorizer = None


def preprocess_text(text):
    """
    Preprocess input text for model prediction
    
    Args:
        text (str): Raw input text
        
    Returns:
        str: Cleaned and preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Handle common greetings and innocent text patterns first
    # Preserve common friendly greetings
    text = re.sub(r'\bh+i+\b', 'hi', text)  # Normalize multiple i's in greetings like "hii", "hiii"
    text = re.sub(r'\bh+e+l+o+\b', 'hello', text)  # Normalize "heelllooo" -> "hello"
    text = re.sub(r'\by+o+\b', 'yo', text)  # Normalize "yyoo" -> "yo"
    
    # Expand contractions to capture more toxic patterns
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'d", " would", text)
    
    # Handle common toxic abbreviations
    text = re.sub(r"\bu\b", "you", text)
    text = re.sub(r"\bur\b", "your", text)
    text = re.sub(r"\bcant\b", "cannot", text)
    text = re.sub(r"\bdont\b", "do not", text)
    text = re.sub(r"\bdoesnt\b", "does not", text)
    text = re.sub(r"\bwont\b", "will not", text)
    
    # Capture repeated characters that may indicate emphasis in toxic comments
    # But be more selective - only normalize if there are 3+ repetitions
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)  # Replace triple+ repetitions with double
    
    # Keep important punctuation that might be relevant for toxicity detection
    # But remove most special characters and digits
    text = re.sub(r'[^a-zA-Z\s!?.]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def get_demo_prediction(text):
    """
    Generate demo predictions when model is not available
    Uses simple keyword matching for demonstration purposes
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Prediction results
    """
    text_lower = text.lower()
    
    # Simple keyword-based detection for demo
    toxic_keywords = ['hate', 'stupid', 'idiot', 'kill', 'die', 'worst']
    severe_keywords = ['kill', 'die', 'murder']
    obscene_keywords = ['damn', 'hell']
    threat_keywords = ['kill', 'hurt', 'attack', 'destroy']
    insult_keywords = ['stupid', 'idiot', 'fool', 'dumb']
    hate_keywords = ['hate']
    
    # Calculate simple scores based on keyword presence
    predictions = {
        'toxic': 0.8 if any(word in text_lower for word in toxic_keywords) else 0.1,
        'severe_toxic': 0.7 if any(word in text_lower for word in severe_keywords) else 0.05,
        'obscene': 0.6 if any(word in text_lower for word in obscene_keywords) else 0.08,
        'threat': 0.75 if any(word in text_lower for word in threat_keywords) else 0.04,
        'insult': 0.65 if any(word in text_lower for word in insult_keywords) else 0.07,
        'identity_hate': 0.6 if any(word in text_lower for word in hate_keywords) else 0.03
    }
    
    return predictions


def predict_with_gemini(text, api_key):
    """
    Predict toxicity using Gemini API (new SDK)
    """
    if genai is None:
        raise Exception("google-genai package is not installed")
        
    try:
        client = genai.Client(api_key=api_key)
        
        prompt = f"""
        Analyze the following text for toxicity and classify it into 6 categories: 
        toxic, severe_toxic, obscene, threat, insult, identity_hate.
        
        For each category, provide a probability score between 0.0 and 1.0.
        Return ONLY a raw JSON object with the category names as keys and the scores as values.
        Do not use markdown formatting or code blocks.
        
        Text to analyze: "{text}"
        """
        
        # Dynamically fetch available models
        models_to_try = []
        try:
            # Look for models that support generateContent
            for m in client.models.list():
                if 'generateContent' in m.supported_actions:
                    models_to_try.append(m.name)
            
            # Prioritize flash models for speed if available
            models_to_try.sort(key=lambda x: ('flash' not in x, x))
            
            if not models_to_try:
                models_to_try = ['gemini-2.0-flash', 'gemini-1.5-flash'] # Fallback
        except Exception as e:
            # If listing fails, fallback to standard names
            print(f"Warning: Failed to list models: {e}")
            models_to_try = ['gemini-2.5-flash', 'gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-pro']
            
        response = None
        last_error = None
        
        for model_name in models_to_try:
            try:
                # the genai client handles the "models/" prefix automatically sometimes, 
                # but if m.name already has it, we just pass it
                clean_name = model_name.replace('models/', '')
                response = client.models.generate_content(
                    model=clean_name,
                    contents=prompt
                )
                break
            except Exception as e:
                last_error = str(e)
                continue
                
        if response is None:
            raise Exception(f"All models failed. Last error: {last_error}. Models tried: {models_to_try}")
            
        response_text = response.text.strip()
        
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
            
        result = json.loads(response_text.strip())
        
        final_result = {}
        for label in TOXICITY_LABELS:
            final_result[label] = float(result.get(label, 0.0))
            
        return final_result
    except Exception as e:
        error_msg = f"Gemini API error: {str(e)}"
        print(error_msg)
        with open("gemini_debug.log", "a") as f:
            f.write(error_msg + "\n")
        return None


def predict_toxicity(text):
    """
    Predict toxicity levels for input text
    
    Args:
        text (str): Input comment text
        
    Returns:
        dict: Dictionary containing predictions for each toxicity category
    """
    # Preprocess the text
    cleaned_text = preprocess_text(text)
    
    # If model is available, use it for prediction
    if model is not None and vectorizer is not None:
        try:
            # Vectorize the text
            text_vectorized = vectorizer.transform([cleaned_text])
            
            # Check if we have any features - if not, it's likely a simple greeting
            dense_vec = text_vectorized.toarray()[0]
            non_zero_features = sum(dense_vec > 0)
            
            # If no features found, treat as very low toxicity (likely a greeting)
            if non_zero_features == 0:
                # Return low toxicity scores for simple greetings
                return {
                    'toxic': 0.1,
                    'severe_toxic': 0.05,
                    'obscene': 0.05,
                    'threat': 0.05,
                    'insult': 0.1,
                    'identity_hate': 0.05
                }
            
            # Get predictions (probabilities for better results)
            predictions = model.predict_proba(text_vectorized)[0]
            
            # Create result dictionary
            result = {
                label: float(pred) 
                for label, pred in zip(TOXICITY_LABELS, predictions)
            }
            
            return result
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            return get_demo_prediction(text)
    else:
        # Use demo prediction
        return get_demo_prediction(text)


@app.route('/')
def home():
    """
    Health check endpoint
    """
    return jsonify({
        'status': 'online',
        'service': 'Toxic Comments Classification API',
        'version': '1.0.0',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts POST requests with text and returns toxicity predictions
    
    Expected JSON format:
    {
        "text": "comment text here"
    }
    
    Returns:
    {
        "success": true,
        "text": "original text",
        "predictions": {...},
        "is_toxic": boolean,
        "max_toxicity": float,
        "timestamp": ISO datetime
    }
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'No text provided. Please send JSON with "text" field.'
            }), 400
        
        text = data['text']
        gemini_api_key = data.get('gemini_api_key', '').strip()
        
        # Validate text is not empty
        if not text or len(text.strip()) == 0:
            return jsonify({
                'success': False,
                'error': 'Text cannot be empty.'
            }), 400
        
        used_gemini = False
        predictions = None
        
        # Try Gemini if API key is provided
        gemini_error = None
        if gemini_api_key:
            try:
                predictions = predict_with_gemini(text, gemini_api_key)
                if predictions is not None:
                    used_gemini = True
                else:
                    gemini_error = "predict_with_gemini returned None"
            except Exception as e:
                gemini_error = str(e)
                
        # Fallback to local model if Gemini failed or no API key
        if predictions is None:
            predictions = predict_toxicity(text)
        
        # Determine if comment is toxic (slightly higher threshold to reduce false positives)
        is_toxic = any(score > 0.65 for score in predictions.values())
        max_toxicity = max(predictions.values())
        
        # Prepare response
        response = {
            'success': True,
            'text': text,
            'predictions': predictions,
            'is_toxic': is_toxic,
            'max_toxicity': round(max_toxicity, 4),
            'toxicity_level': get_toxicity_level(max_toxicity),
            'timestamp': datetime.now().isoformat(),
            'demo_mode': model is None and not used_gemini,
            'used_gemini': used_gemini,
            'gemini_error': gemini_error
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple texts
    
    Expected JSON format:
    {
        "texts": ["comment 1", "comment 2", ...]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'success': False,
                'error': 'No texts provided. Please send JSON with "texts" array.'
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'success': False,
                'error': 'Texts must be an array.'
            }), 400
        
        # Limit batch size
        if len(texts) > 100:
            return jsonify({
                'success': False,
                'error': 'Maximum batch size is 100 texts.'
            }), 400
        
        # Process each text
        results = []
        for text in texts:
            if text and len(text.strip()) > 0:
                predictions = predict_toxicity(text)
                is_toxic = any(score > 0.5 for score in predictions.values())
                max_toxicity = max(predictions.values())
                
                results.append({
                    'text': text,
                    'predictions': predictions,
                    'is_toxic': is_toxic,
                    'max_toxicity': round(max_toxicity, 4)
                })
        
        return jsonify({
            'success': True,
            'count': len(results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'An error occurred: {str(e)}'
        }), 500


def get_toxicity_level(score):
    """
    Categorize toxicity score into levels
    
    Args:
        score (float): Toxicity score between 0 and 1
        
    Returns:
        str: Toxicity level description
    """
    if score < 0.2:
        return 'Safe'
    elif score < 0.4:
        return 'Moderate'
    elif score < 0.6:
        return 'Toxic'
    else:
        return 'Highly Toxic'


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """
    Get API statistics and model information
    """
    return jsonify({
        'model_loaded': model is not None,
        'categories': TOXICITY_LABELS,
        'demo_mode': model is None,
        'endpoints': {
            'predict': '/api/predict',
            'batch_predict': '/api/batch-predict',
            'stats': '/api/stats'
        }
    })


# Error handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs(os.path.join(os.path.dirname(__file__), 'models'), exist_ok=True)
    
    # Load models on startup
    load_models()
    
    # Run the Flask app
    print("\n" + "="*50)
    print(">>> Toxic Comments Classification API")
    print("="*50)
    print(f"Server starting on http://localhost:5000")
    print("="*50 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
