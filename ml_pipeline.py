import pandas as pd
import numpy as np
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.cluster import DBSCAN
import joblib
import os

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def generate_sample_data(n_samples=2000):
    np.random.seed(42)
    # Generate mock locations (e.g., around a city center)
    base_lat, base_lng = 40.7128, -74.0060 # NYC coordinates
    lats = base_lat + np.random.normal(0, 0.05, n_samples)
    lngs = base_lng + np.random.normal(0, 0.05, n_samples)
    
    hours = np.random.randint(0, 24, n_samples)
    weather = np.random.choice(['Clear', 'Rain', 'Snow', 'Fog'], n_samples, p=[0.6, 0.2, 0.1, 0.1])
    road_type = np.random.choice(['Highway', 'Urban', 'Rural'], n_samples, p=[0.4, 0.5, 0.1])
    
    # Introduce some logical rules to make risk predictable
    risk_prob = np.zeros(n_samples)
    
    for i in range(n_samples):
        prob = 0.1 # Base risk
        # Night time increases risk
        if hours[i] >= 20 or hours[i] <= 5:
            prob += 0.3
        # Weather increases risk
        if weather[i] == 'Rain':
            prob += 0.2
        elif weather[i] in ['Snow', 'Fog']:
            prob += 0.3
        # Rural roads at night are dangerous (interaction)
        if road_type[i] == 'Rural' and (hours[i] >= 20 or hours[i] <= 5):
            prob += 0.2
        
        # Spatial hotspot 1
        dist_to_hotspot1 = np.sqrt((lats[i] - 40.73)**2 + (lngs[i] - -73.99)**2)
        if dist_to_hotspot1 < 0.01:
            prob += 0.4
            
        risk_prob[i] = min(prob, 0.95)
        
    severity = np.random.binomial(1, risk_prob)
    
    df = pd.DataFrame({
        'lat': lats,
        'lng': lngs,
        'hour_of_day': hours,
        'weather': weather,
        'road_type': road_type,
        'high_severity': severity
    })
    
    return df

def train_and_evaluate(df):
    X = df[['lat', 'lng', 'hour_of_day', 'weather', 'road_type']]
    y = df['high_severity']
    
    # Simple train-test split based on index for demo
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Train EBM
    ebm = ExplainableBoostingClassifier(interactions=3, random_state=42)
    ebm.fit(X_train, y_train)
    
    # Train Baseline (Random Forest) with OneHotEncoding
    from sklearn.pipeline import Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['weather', 'road_type'])
        ],
        remainder='passthrough'
    )
    rf = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf.fit(X_train, y_train)
    
    # Evaluate
    ebm_preds = ebm.predict(X_test)
    ebm_probs = ebm.predict_proba(X_test)[:, 1]
    
    rf_preds = rf.predict(X_test)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    
    metrics = {
        'EBM': {
            'Accuracy': accuracy_score(y_test, ebm_preds),
            'F1': f1_score(y_test, ebm_preds),
            'AUC': roc_auc_score(y_test, ebm_probs)
        },
        'Baseline (RF)': {
            'Accuracy': accuracy_score(y_test, rf_preds),
            'F1': f1_score(y_test, rf_preds),
            'AUC': roc_auc_score(y_test, rf_probs)
        }
    }
    
    # Save model
    joblib.dump(ebm, os.path.join(MODEL_DIR, "ebm_model.pkl"))
    
    return ebm, metrics, X_test, y_test

def perform_clustering(df):
    # Only cluster high severity accidents
    high_risk = df[df['high_severity'] == 1].copy()
    coords = high_risk[['lat', 'lng']].values
    
    # DBSCAN using haversine metric (requires radians)
    kms_per_radian = 6371.0088
    epsilon = 1.0 / kms_per_radian # 1km radius
    
    db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine')
    clusters = db.fit_predict(np.radians(coords))
    
    high_risk['cluster'] = clusters
    return high_risk
