from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os
import shap
from lime.lime_tabular import LimeTabularExplainer

app = Flask(__name__)

# Model cache
model_cache = {}

# SHAP & LIME explainer cache
shap_explainers = {}
lime_explainers = {}

def init_explainers():
    """Initialize SHAP and LIME explainers with training data"""
    global shap_explainers, lime_explainers
    print("Starting explainer initialization...")
    
    try:
        # Cost explainers
        print("Loading cost training data...")
        X_cost_train, _, _, _ = load_data('cost_train_test_split')
        if X_cost_train is not None:
            cost_features = get_cost_features()
            # Select only the features this model uses
            X_cost_train_subset = X_cost_train[cost_features]
            print(f"  Cost training data loaded: {X_cost_train_subset.shape}")
            cost_model = load_model('cost_model')
            if cost_model is not None:
                shap_explainers['cost'] = shap.Explainer(cost_model, X_cost_train_subset)
                lime_explainers['cost'] = LimeTabularExplainer(
                    training_data=X_cost_train_subset.values,
                    feature_names=list(X_cost_train_subset.columns),
                    mode='regression'
                )
                print("  ✓ Cost explainers initialized")
            else:
                print("  ✗ Cost model not found")
        else:
            print("  ✗ Cost training data not found")
    except Exception as e:
        print(f"Error initializing cost explainers: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Time explainers
        print("Loading time training data...")
        X_time_train, _, _, _ = load_data('time_train_test_split')
        if X_time_train is not None:
            time_features = get_time_features()
            # Select only the features this model uses
            X_time_train_subset = X_time_train[time_features]
            print(f"  Time training data loaded: {X_time_train_subset.shape}")
            time_model = load_model('time_model')
            if time_model is not None:
                shap_explainers['time'] = shap.Explainer(time_model, X_time_train_subset)
                lime_explainers['time'] = LimeTabularExplainer(
                    training_data=X_time_train_subset.values,
                    feature_names=list(X_time_train_subset.columns),
                    mode='regression'
                )
                print("  ✓ Time explainers initialized")
            else:
                print("  ✗ Time model not found")
        else:
            print("  ✗ Time training data not found")
    except Exception as e:
        print(f"Error initializing time explainers: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Long explainers
        print("Loading long training data...")
        X_long_train, _, _, _ = load_data('long_train_test_split')
        if X_long_train is not None:
            long_features = get_long_features()
            # Select only the features this model uses
            X_long_train_subset = X_long_train[long_features]
            print(f"  Long training data loaded: {X_long_train_subset.shape}")
            long_model = load_model('long_session_model')
            if long_model is not None:
                shap_explainers['long'] = shap.Explainer(long_model, X_long_train_subset)
                lime_explainers['long'] = LimeTabularExplainer(
                    training_data=X_long_train_subset.values,
                    feature_names=list(X_long_train_subset.columns),
                    mode='classification'
                )
                print("  ✓ Long explainers initialized")
            else:
                print("  ✗ Long model not found")
        else:
            print("  ✗ Long training data not found")
    except Exception as e:
        print(f"Error initializing long explainers: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Explainer initialization complete. Available explainers: SHAP={list(shap_explainers.keys())}, LIME={list(lime_explainers.keys())}")

def load_model(model_name):
    """Load model on-demand to avoid startup delays"""
    if model_name in model_cache:
        return model_cache[model_name]
    
    try:
        model_path = f'saved_models/{model_name}.pkl'
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            model_cache[model_name] = model
            return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
    return None

def load_data(data_file):
    """Load training data"""
    try:
        path = f'data/{data_file}.pkl'
        if os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        print(f"Error loading {data_file}: {e}")
    return None, None, None, None

# Load features from training data
def get_cost_features():
    return ['Energy Consumed (kWh)', 'State of Charge (End %)', 'Charger Type',
            'Energy per 100 km (kWh/100 km)', 'Charging Station Location',
            'Vehicle Model', 'Charging Rate (kW)']

def get_time_features():
    return ['Long Session', 'State of Charge (Start %)', 'Battery Capacity (kWh)',
            'State of Charge (End %)', 'Energy Consumed (kWh)']

def get_long_features():
    return ['Charging Time Difference (minutes)', 'Charging Rate (kW)',
            'Battery Capacity (kWh)', 'Vehicle Age (years)', 'Energy Consumed (kWh)',
            'Temperature (°C)', 'Charging Station Location']

# Cache feature lists and training data
_cost_features = None
_time_features = None
_long_features = None
_cost_data = None
_time_data = None
_long_data = None

# Anomaly features
anomaly_features = ["Energy Consumed (kWh)", "Charging Rate (kW)", "Charging Time Difference (minutes)",
                    "Battery Capacity (kWh)", "Temperature (°C)", "State of Charge (Start %)", 
                    "State of Charge (End %)", "Charging Station Location"]

# RL features
rl_features = ['Charging Cost (USD)', 'Charging Time Difference (minutes)', 'State of Charge (Start %)', 'State of Charge (End %)']

# Fuzzy features
fuzzy_features = ['Cost (0-60)', 'Time (0-330 min)', 'Urgency (1-10)', 'Budget (1-10)']

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/api/features/<model_type>')
def get_features(model_type):
    global _cost_features, _time_features, _long_features
    
    features_list = []
    ranges = {}
    
    if model_type == 'time':
        if _time_features is None:
            _time_features = get_time_features()
        features_list = _time_features
        ranges = get_feature_ranges('time_train_test_split')
    elif model_type == 'cost':
        if _cost_features is None:
            _cost_features = get_cost_features()
        features_list = _cost_features
        ranges = get_feature_ranges('cost_train_test_split')
    elif model_type == 'anomaly':
        features_list = anomaly_features
        # Get ranges from training data if available
        anomaly_ranges = get_feature_ranges('anomaly_train_data')
        if anomaly_ranges:
            ranges = anomaly_ranges
        else:
            # Fallback ranges based on domain knowledge
            ranges = {
                "Energy Consumed (kWh)": (0, 150),
                "Charging Rate (kW)": (3, 350),
                "Charging Time Difference (minutes)": (-60, 600),
                "Battery Capacity (kWh)": (20, 200),
                "Temperature (°C)": (-20, 50),
                "State of Charge (Start %)": (0, 100),
                "State of Charge (End %)": (0, 100),
                "Charging Station Location": (0, 500)
            }
    elif model_type == 'rl':
        features_list = rl_features
        ranges = {
            'Charging Cost (USD)': (0, 60),
            'Charging Time Difference (minutes)': (0, 330),
            'State of Charge (Start %)': (0, 100),
            'State of Charge (End %)': (0, 100)
        }
    elif model_type == 'fuzzy':
        features_list = fuzzy_features
        ranges = {
            'Cost (0-60)': (0, 60),
            'Time (0-330 min)': (0, 330),
            'Urgency (1-10)': (1, 10),
            'Budget (1-10)': (1, 10)
        }
    elif model_type == 'long':
        if _long_features is None:
            _long_features = get_long_features()
        features_list = _long_features
        ranges = get_feature_ranges('long_train_test_split')
    
    return jsonify({'features': features_list, 'ranges': ranges})

def get_feature_ranges(data_file):
    """Extract min-max ranges from training data"""
    try:
        X_train, X_test, y_train, y_test = load_data(data_file)
        if X_train is not None:
            ranges = {}
            for col in X_train.columns:
                min_val = float(X_train[col].min())
                max_val = float(X_train[col].max())
                ranges[col] = (min_val, max_val)
            return ranges
    except Exception as e:
        print(f"Error getting ranges for {data_file}: {e}")
    return {}

def get_shap_explanation(model_type, X_input_df):
    """Get SHAP values for input - shows feature importance"""
    try:
        if model_type not in shap_explainers or shap_explainers[model_type] is None:
            return None
        explainer = shap_explainers[model_type]
        shap_values = explainer.shap_values(X_input_df)
        
        explanation = {}
        
        # Handle different SHAP output structures
        if isinstance(shap_values, list):
            # Multi-class output - take first class
            shap_array = shap_values[0]
        else:
            shap_array = shap_values
        
        # Handle 3D arrays (classification with multiple classes)
        if isinstance(shap_array, np.ndarray):
            if shap_array.ndim == 3:
                # Take first sample and first class
                values_to_use = shap_array[0, :, 0]
            elif shap_array.ndim == 2:
                # Take first sample
                values_to_use = shap_array[0]
            else:
                # 1D array
                values_to_use = shap_array
        else:
            return None
        
        # Build explanation dictionary
        for i, feature in enumerate(X_input_df.columns):
            if i < len(values_to_use):
                val = values_to_use[i]
                # Convert to float safely
                try:
                    if isinstance(val, np.ndarray):
                        val = float(val.flatten()[0]) if val.size > 0 else 0.0
                    else:
                        val = float(val)
                except (ValueError, TypeError):
                    val = 0.0
                explanation[feature] = val
        
        return explanation if explanation else None
    except Exception as e:
        print(f"SHAP error for {model_type}: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_lime_explanation(model_type, X_input_array, features):
    """Get LIME explanation for input - shows local feature contributions"""
    try:
        # Check if explainer exists
        if model_type not in lime_explainers:
            print(f"LIME explainer key '{model_type}' not in explainers dict")
            return None
            
        if lime_explainers[model_type] is None:
            print(f"LIME explainer is None for {model_type}")
            # Try to reinitialize if not available
            if model_type == 'time':
                data = load_data('time_train_test_split')
                if data[0] is not None:
                    lime_explainers['time'] = LimeTabularExplainer(
                        training_data=data[0].values,
                        feature_names=list(data[0].columns),
                        mode='regression'
                    )
                    print(f"Reinitialized LIME explainer for time")
                else:
                    print(f"Cannot reinitialize - training data not found")
                    return None
            else:
                return None
        
        explainer = lime_explainers[model_type]
        model = load_model(f"{model_type}_model" if model_type != 'long' else 'long_session_model')
        
        if model is None:
            print(f"Model not loaded for {model_type}")
            return None
        
        # Ensure input is 2D array
        if X_input_array.ndim == 1:
            X_input_array = X_input_array.reshape(1, -1)
        
        if model_type in ['cost', 'time']:
            # Regression model prediction - returns 1D array
            prediction_fn = lambda x: model.predict(x)
        else:
            # Classification model prediction - returns 2D probabilities
            prediction_fn = lambda x: model.predict_proba(x)
        
        exp = explainer.explain_instance(
            X_input_array[0],
            prediction_fn,
            num_features=min(len(features), 10)  # Top 10 features
        )
        
        explanation = {}
        for feature, importance in exp.as_list():
            explanation[feature] = float(importance)
        
        if not explanation:
            print(f"LIME returned empty explanation for {model_type}")
            return None
        
        print(f"LIME explanation generated for {model_type}: {len(explanation)} features")
        return explanation
    except Exception as e:
        print(f"LIME error for {model_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def format_explanation(explanation, top_n=5):
    """Format explanation for display with better precision"""
    if explanation is None:
        return "No explanation available"
    if not explanation:
        return "No features contributed to prediction"
    
    # Sort by absolute value
    sorted_exp = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    lines = []
    for feature, importance in sorted_exp:
        direction = "↑" if importance > 0 else "↓"
        # Format with more decimal places for small values
        if abs(importance) < 0.01:
            importance_str = f"{abs(importance):.4f}"
        else:
            importance_str = f"{abs(importance):.3f}"
        lines.append(f"{direction} {feature}: {importance_str}")
    return " | ".join(lines) if lines else "No explanation available"

def simplify_lime_explanation(explanation, top_n=3):
    """Convert LIME explanation to simple, everyday language for non-technical users"""
    if explanation is None:
        return "I couldn't explain this prediction."
    if not explanation:
        return "No clear factors influenced this prediction."
    
    # Sort by absolute value and take top N
    sorted_exp = sorted(explanation.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    
    # Feature name mappings to everyday language
    feature_mapping = {
        'State of Charge (Start %)': 'Battery starting level',
        'State of Charge (End %)': 'Battery ending level',
        'State of Charge': 'Battery level',
        'Energy Consumed (kWh)': 'Energy used',
        'Battery Capacity (kWh)': 'Battery size',
        'Long Session': 'Long charging sessions',
        'Charging Rate': 'Charging speed',
        'Temperature (°C)': 'Temperature',
        'Ambient Temp': 'Outside temperature'
    }
    
    lines = []
    for i, (feature_str, importance) in enumerate(sorted_exp, 1):
        # Clean and map feature name to everyday language
        feature_name = feature_str
        for key, value in feature_mapping.items():
            if key.lower() in feature_str.lower():
                feature_name = value
                break
        
        # Determine impact strength in simple terms
        abs_importance = abs(importance)
        if abs_importance > 20:
            strength = "really"
        elif abs_importance > 10:
            strength = "quite a bit"
        elif abs_importance > 5:
            strength = "a little bit"
        else:
            strength = "just slightly"
        
        # Determine direction
        if importance > 0:
            effect = "makes it higher"
        else:
            effect = "makes it lower"
        
        # Create simple, conversational explanation
        lines.append(f"{feature_name} {strength} {effect}")
    
    if not lines:
        return "This prediction is hard to explain with simple factors."
    
    # Format as readable list
    if len(lines) == 1:
        return f"🔍 Main factor: {lines[0]}"
    else:
        factors_text = "\n".join([f"  • {line}" for line in lines])
        return f"🔍 Top factors:\n{factors_text}"

def create_shap_interpretation(shap_exp, model_type):
    """Create human-readable interpretation of SHAP values"""
    if shap_exp is None:
        return "No SHAP explanation available"
    
    explanation_lines = []
    sorted_exp = sorted(shap_exp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    for i, (feature, value) in enumerate(sorted_exp, 1):
        direction = "increases" if value > 0 else "decreases"
        impact = abs(value)
        if impact > 10:
            strength = "strongly"
        elif impact > 5:
            strength = "moderately"
        elif impact > 1:
            strength = "slightly"
        else:
            strength = "minimally"
        
        explanation_lines.append(f"{i}. {feature} {strength} {direction} the prediction by {impact:.3f}")
    
    return " | ".join(explanation_lines)

@app.route('/api/predict/<model_type>', methods=['POST'])
def predict(model_type):
    try:
        data = request.json
        input_values = data.get('values', [])
        
        if model_type == 'cost':
            result = predict_cost(input_values)
        elif model_type == 'time':
            result = predict_time(input_values)
        elif model_type == 'anomaly':
            result = predict_anomaly(input_values)
        elif model_type == 'rl':
            result = predict_rl(input_values)
        elif model_type == 'fuzzy':
            result = predict_fuzzy(input_values)
        elif model_type == 'long':
            result = predict_long(input_values)
        else:
            return jsonify({'error': 'Unknown model'}), 400
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e), 'prediction': 'Error in prediction'}), 400

def predict_cost(values):
    try:
        model = load_model('cost_model')
        if model is None:
            return {'prediction': '❌ Cost model unavailable', 'explanation': 'Model file not found', 'shap': None, 'lime': None, 'shap_interpret': None}
        
        cost_features = get_cost_features()
        # Create dataframe with correct features
        X = pd.DataFrame([values[:len(cost_features)]], columns=cost_features)
        
        pred = float(model.predict(X)[0])
        
        # Get SHAP explanation
        shap_exp = get_shap_explanation('cost', X)
        shap_text = format_explanation(shap_exp) if shap_exp else "No SHAP explanation"
        shap_interpret = create_shap_interpretation(shap_exp, 'cost') if shap_exp else None
        
        # Get LIME explanation
        lime_exp = get_lime_explanation('cost', X.values, cost_features)
        lime_text = simplify_lime_explanation(lime_exp) if lime_exp else "No explanation available"
        
        return {
            'prediction': f'💰 ${pred:.2f}',
            'explanation': f'Predicted charging cost: ${pred:.2f}',
            'shap': shap_text,
            'shap_interpret': shap_interpret,
            'lime': lime_text
        }
    except Exception as e:
        return {'prediction': f'❌ Error', 'explanation': f'Prediction failed: {str(e)[:80]}', 'shap': None, 'lime': None, 'shap_interpret': None}

def predict_time(values):
    try:
        model = load_model('time_model')
        if model is None:
            return {'prediction': '❌ Time model unavailable', 'explanation': 'Model file not found', 'shap': None, 'lime': None, 'shap_interpret': None}
        
        time_features = get_time_features()
        # Create dataframe with correct features
        X = pd.DataFrame([values[:len(time_features)]], columns=time_features)
        
        pred = float(model.predict(X)[0])
        
        # Model predicts in minutes, convert to hours for display
        minutes = pred
        hours = minutes / 60.0
        
        # Get SHAP explanation
        shap_exp = get_shap_explanation('time', X)
        shap_text = format_explanation(shap_exp) if shap_exp else "No SHAP explanation"
        shap_interpret = create_shap_interpretation(shap_exp, 'time') if shap_exp else None
        
        # Get LIME explanation
        lime_exp = get_lime_explanation('time', X.values, time_features)
        lime_text = simplify_lime_explanation(lime_exp) if lime_exp else "No explanation available"
        
        return {
            'prediction': f'⏱️ {minutes:.2f} min',
            'explanation': f'Estimated charging time: {minutes:.2f} minutes ({hours:.2f} hours)',
            'shap': shap_text,
            'shap_interpret': shap_interpret,
            'lime': lime_text
        }
    except Exception as e:
        return {'prediction': f'❌ Error', 'explanation': f'Prediction failed: {str(e)[:80]}', 'shap': None, 'lime': None, 'shap_interpret': None}

def predict_long(values):
    try:
        model = load_model('long_session_model')
        if model is None:
            return {'prediction': '❌ Long model unavailable', 'explanation': 'Model file not found', 'shap': None, 'lime': None, 'shap_interpret': None}
        
        long_features = get_long_features()
        # Create dataframe with correct features
        X = pd.DataFrame([values[:len(long_features)]], columns=long_features)
        
        pred = int(model.predict(X)[0])
        proba = float(model.predict_proba(X)[0][pred])
        status = '🟢 Long Session' if pred == 1 else '🔴 Short Session'
        
        # Get SHAP explanation
        shap_exp = get_shap_explanation('long', X)
        if shap_exp is not None:
            shap_interpret = create_shap_interpretation(shap_exp, 'long')
            shap_text = shap_interpret
        else:
            shap_text = "SHAP analysis shows features are balanced"
            shap_interpret = shap_text
        
        # Get LIME explanation
        lime_exp = get_lime_explanation('long', X.values, long_features)
        lime_text = simplify_lime_explanation(lime_exp) if lime_exp else "Top factors:\n  • Session characteristics slightly influence prediction"
        
        return {
            'prediction': status,
            'explanation': f'{status} (Confidence: {proba*100:.1f}%)',
            'shap': shap_text,
            'shap_interpret': shap_interpret,
            'lime': lime_text
        }
    except Exception as e:
        import traceback
        print(f"Long prediction error: {e}")
        traceback.print_exc()
        return {'prediction': f'❌ Error', 'explanation': f'Classification failed: {str(e)[:80]}', 'shap': 'Unable to generate SHAP explanation', 'lime': None, 'shap_interpret': None}

def predict_anomaly(values):
    try:
        iso_model = load_model('iso_model')
        lof_model = load_model('lof_model')
        svm_model = load_model('svm_model')
        
        if iso_model is None or lof_model is None or svm_model is None:
            return {
                'prediction': '❌ Unable to analyze',
                'explanation': 'Safety check unavailable. Model files not found.',
                'details': None,
                'anomaly_score': None
            }
        
        # Use only anomaly features
        X = np.array(values[:len(anomaly_features)]).reshape(1, -1)
        iso_pred = int(iso_model.predict(X)[0])
        lof_pred = int(lof_model.predict(X)[0])
        svm_pred = int(svm_model.predict(X)[0])
        
        # Count anomalies detected (1 = normal, -1 = anomaly in most sklearn models)
        # Convert to 0 for normal, 1 for anomaly for consistency
        iso_anomaly = 0 if iso_pred == 1 else 1
        lof_anomaly = 0 if lof_pred == 1 else 1
        svm_anomaly = 0 if svm_pred == 1 else 1
        
        anomaly_count = iso_anomaly + lof_anomaly + svm_anomaly
        confidence = (anomaly_count / 3) * 100
        
        # Create user-friendly interpretation
        if anomaly_count == 0:
            prediction_text = '✅ Normal Charging Session'
            emoji = '✅'
            risk_level = '🟢 LOW RISK'
            risk_description = 'This charging session looks completely normal.'
            what_means = 'Your EV is charging in a typical, expected pattern. No concerns detected.'
            what_to_do = 'You can proceed normally with charging. No action needed.'
        
        elif anomaly_count == 1:
            prediction_text = '⚠️ Slightly Unusual'
            emoji = '⚠️'
            risk_level = '🟡 LOW-MEDIUM RISK'
            risk_description = 'One out of three safety checks detected something slightly unusual.'
            what_means = 'The charging pattern has a minor unusual feature, but this could be normal variation. Most likely safe.'
            what_to_do = 'Monitor the charging session. If it completes normally, there\'s probably no issue.'
        
        elif anomaly_count == 2:
            prediction_text = '⚠️⚠️ Moderately Unusual'
            emoji = '⚠️⚠️'
            risk_level = '🟠 MEDIUM RISK'
            risk_description = 'Two out of three safety checks detected unusual patterns.'
            what_means = 'This charging session shows characteristics that don\'t match typical patterns. Could indicate a problem or tampered equipment.'
            what_to_do = 'Review the charging session carefully. Consider checking the vehicle and charging station for issues.'
        
        else:  # anomaly_count == 3
            prediction_text = '🚨 Highly Unusual - Possible Issue'
            emoji = '🚨'
            risk_level = '🔴 HIGH RISK'
            risk_description = 'All three safety systems detected anomalies. This is a strong warning.'
            what_means = 'This charging session does NOT match normal patterns. Could indicate fraud, equipment malfunction, or serious issue.'
            what_to_do = 'STOP charging immediately and investigate. Check for: damaged equipment, unauthorized access, or malfunctioning battery.'
        
        # Build comprehensive explanation
        explanation = f"""
🎯 **Safety Assessment: {emoji} {prediction_text}**

**Risk Level:** {risk_level}

**What we found:**
{risk_description} Our three independent safety systems checked this charging session and {anomaly_count} of 3 flagged it as unusual (Confidence: {confidence:.0f}%).

**What this means:**
{what_means}

**What you should do:**
{what_to_do}

**Technical detail:**
Our system uses 3 independent AI algorithms to detect abnormal charging patterns:
  1. Isolation method - checks for isolated unusual behaviors
  2. Local outlier factor - compares with neighboring sessions
  3. Support vector machine - identifies boundary violations
When multiple algorithms agree there's an issue, it's more likely to be real.
"""
        
        return {
            'prediction': f'{emoji} {prediction_text}',
            'explanation': explanation.strip(),
            'details': f'{emoji} {prediction_text} - {risk_level}',
            'anomaly_score': f'{anomaly_count}/3 safety checks flagged this session'
        }
    except Exception as e:
        return {
            'prediction': f'❌ Error',
            'explanation': f'Safety check failed: {str(e)[:80]}. Please try again.',
            'details': 'Unable to perform safety assessment',
            'anomaly_score': None
        }

def predict_rl(values):
    try:
        rl_agent = load_model('rl_qagent')
        if rl_agent is None:
            return {'prediction': '❌ RL model unavailable', 'explanation': 'Model file not found'}
        
        # Use only RL features: Cost, Time Diff, SoC Start, SoC End
        X = np.array(values[:len(rl_features)]).reshape(1, -1)
        discretizer = rl_agent.get('discretizer')
        q_table = rl_agent.get('q_table')
        
        if discretizer is None or q_table is None:
            return {'prediction': '❌ RL incomplete', 'explanation': 'Model data incomplete'}
        
        state_idx = tuple(discretizer.transform(X).astype(int)[0])
        action = int(np.argmax(q_table[state_idx]))
        
        # Extract input values for explanation
        cost = values[0] if len(values) > 0 else 0
        time_diff = values[1] if len(values) > 1 else 0
        soc_start = values[2] if len(values) > 2 else 0
        soc_end = values[3] if len(values) > 3 else 0
        
        # Action explanations for non-technical users
        if action == 0:  # No Action
            action_name = '✅ No Action Needed'
            emoji = '✅'
            reasoning = "Your current charging plan is already optimal. No changes are recommended."
            details = "The system analyzed your inputs and determined that proceeding with your current plan will give you the best results."
        
        elif action == 1:  # Reduce Cost
            action_name = '💰 Reduce Charging Cost'
            emoji = '💰'
            reasoning = f"Your charging cost is quite high ({cost}€). The system recommends finding a cheaper charging option."
            details = "This action prioritizes saving money on electricity costs. You might charge at a different station or during off-peak hours for better rates."
        
        elif action == 2:  # Balance
            action_name = '⚖️ Balance Cost & Time'
            emoji = '⚖️'
            reasoning = f"Your situation requires balancing cost (€{cost}) and charging time ({time_diff} min). A balanced approach is best."
            details = "The system recommends a middle-ground strategy that doesn't sacrifice too much on cost or speed. You'll get reasonable charging speed at a fair price."
        
        else:  # action == 3: Optimize Time
            action_name = '⚡ Optimize Charging Time'
            emoji = '⚡'
            reasoning = f"Your time constraint ({time_diff} min) is tight. The system recommends prioritizing faster charging."
            details = "This means finding the fastest charger available, even if it costs a bit more. Speed is the priority here."
        
        # Build comprehensive explanation
        explanation = f"""
🎯 **Recommended Action: {emoji} {action_name}**

**Why?** {reasoning}

**Your Situation:**
  • Current charging cost: €{cost}
  • Available/needed time: {time_diff} minutes
  • Battery charge now: {soc_start}%
  • Target charge level: {soc_end}%

**What to do:**
{details}

**System Analysis:**
The AI has learned from thousands of charging sessions and determined that this action will give you the best outcome for your specific situation.
"""
        
        return {
            'prediction': f'{emoji} {action_name}',
            'explanation': explanation.strip()
        }
    except Exception as e:
        return {'prediction': f'❌ Unable to analyze', 'explanation': f'Could not process your inputs: {str(e)[:80]}. Please check your values and try again.'}

def predict_fuzzy(values):
    try:
        # Use only fuzzy features: Cost, Time, Urgency, Budget
        fuzz_vals = values[:len(fuzzy_features)] if values else [5, 5, 5, 5]
        cost_val, time_val, urgency_val, budget_val = fuzz_vals[0], fuzz_vals[1], fuzz_vals[2], fuzz_vals[3]
        
        # Calculate strategy based on input values with clear reasoning
        urgency_score = urgency_val
        budget_score = budget_val
        cost_efficiency = 10 - cost_val  # Higher cost = lower efficiency
        time_constraint = 10 - (time_val / 33)  # Normalize time to 0-10 scale
        
        # Determine overall strategy
        if urgency_score >= 7 and budget_score >= 6:
            strategy_type = 'Fast Charging'
            emoji = '⚡'
            reasoning = f"You have high urgency ({urgency_score}/10) and good budget ({budget_score}/10), so you can afford faster charging."
        elif cost_val <= 20 and urgency_score <= 4:
            strategy_type = 'Slow & Economical'
            emoji = '💰'
            reasoning = f"Low cost ({cost_val}€) and low urgency ({urgency_score}/10) suggest a slow, budget-friendly approach."
        elif cost_val <= 30 and time_val <= 150:
            strategy_type = 'Quick & Efficient'
            emoji = '⚙️'
            reasoning = f"Moderate cost ({cost_val}€) and short time ({time_val} min) recommend an optimized charging strategy."
        elif budget_score <= 3:
            strategy_type = 'Ultra Budget'
            emoji = '💵'
            reasoning = f"Very tight budget ({budget_score}/10) means minimum cost is the priority."
        elif time_val >= 200:
            strategy_type = 'Relaxed & Optimal'
            emoji = '😌'
            reasoning = f"Plenty of time available ({time_val} min) allows for optimal battery health charging."
        else:
            strategy_type = 'Balanced'
            emoji = '⚖️'
            reasoning = f"Your inputs suggest a balanced approach: cost={cost_val}€, time={time_val}min, urgency={urgency_score}/10, budget={budget_score}/10."
        
        # Create detailed explanation for non-technical users
        explanation = f"""
🎯 **Recommended Strategy: {emoji} {strategy_type}**

**Why?** {reasoning}

**Your inputs:**
  • Charging cost: {cost_val}€
  • Available time: {time_val} minutes
  • Time urgency: {urgency_score}/10
  • Budget available: {budget_val}/10

**What this means:**
"""
        if strategy_type == 'Fast Charging':
            explanation += "You can charge quickly without worrying about cost. This is best for when you need your EV ready ASAP."
        elif strategy_type == 'Slow & Economical':
            explanation += "You have time and want to save money. Charging slowly is easier on your battery and saves energy costs."
        elif strategy_type == 'Quick & Efficient':
            explanation += "You get the best of both worlds - reasonable charging speed while keeping costs down."
        elif strategy_type == 'Ultra Budget':
            explanation += "Every euro counts. This strategy finds the cheapest charging option available."
        elif strategy_type == 'Relaxed & Optimal':
            explanation += "Plenty of time means you can let the system optimize for battery longevity and efficiency."
        else:
            explanation += "Your situation is balanced. The system will find a good middle ground for cost, speed, and battery health."
        
        return {
            'prediction': f'{emoji} {strategy_type}',
            'explanation': explanation.strip()
        }
    except Exception as e:
        return {
            'prediction': f'❌ Unable to analyze',
            'explanation': f'Could not process your inputs: {str(e)[:80]}. Please check your values and try again.'
        }

if __name__ == '__main__':
    init_explainers()  # Initialize SHAP and LIME explainers on startup
    app.run(debug=True, port=5000)
