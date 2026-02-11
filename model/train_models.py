# Model training for heart disease prediction
# Using 6 different ML classifiers

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(filepath='heart.csv'):
    """Load dataset and encode categorical columns"""
    df = pd.read_csv(filepath)
    df_processed = df.copy()
    
    # these are the categorical columns that need encoding
    cat_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
    label_encoders = {}
    
    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    X = df_processed.drop('HeartDisease', axis=1)
    y = df_processed['HeartDisease']
    
    return X, y, label_encoders, df


def get_models():
    """Initialize all the classifiers we want to compare"""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }


def calculate_metrics(y_true, y_pred, y_proba=None):
    """Get all the evaluation metrics for a model"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, average='weighted'),
        'Recall': recall_score(y_true, y_pred, average='weighted'),
        'F1 Score': f1_score(y_true, y_pred, average='weighted'),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    
    # AUC needs probability scores
    if y_proba is not None:
        try:
            metrics['AUC'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics


def train_and_evaluate_models(X, y, test_size=0.2, random_state=42):
    """Train all models and collect results"""
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # need to scale for kNN and logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = get_models()
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = model
        
        y_pred = model.predict(X_test_scaled)
        
        # get probabilities if the model supports it
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        results[name] = {
            'metrics': metrics,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_proba
        }
    
    return results, trained_models, scaler, (X_test, y_test)


def get_metrics_dataframe(results):
    """Create a dataframe with all model metrics for easy comparison"""
    data = []
    for name, result in results.items():
        row = {'Model': name}
        row.update(result['metrics'])
        data.append(row)
    
    df = pd.DataFrame(data)
    cols = ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    df = df[cols]
    
    # round to 4 decimal places
    for col in df.columns[1:]:
        df[col] = df[col].round(4)
    
    return df


def get_model_observations(results):
    """Generate observations based on metrics"""
    observations = {}
    
    for model_name, result in results.items():
        m = result['metrics']
        acc, auc, mcc = m['Accuracy'], m['AUC'], m['MCC']
        
        obs = []
        
        # accuracy level
        if acc >= 0.85:
            obs.append(f"High accuracy ({acc:.2%})")
        elif acc >= 0.75:
            obs.append(f"Good accuracy ({acc:.2%})")
        else:
            obs.append(f"Moderate accuracy ({acc:.2%})")
        
        # discrimination ability based on AUC
        if auc >= 0.90:
            obs.append("Excellent discrimination ability")
        elif auc >= 0.80:
            obs.append("Good discrimination ability")
        else:
            obs.append("Fair discrimination ability")
        
        # correlation from MCC
        if mcc >= 0.6:
            obs.append("Strong correlation between predicted and actual")
        elif mcc >= 0.4:
            obs.append("Moderate correlation")
        else:
            obs.append("Weak correlation")
        
        # Model-specific observations
        if model_name == 'Logistic Regression':
            obs.append("Linear decision boundary, good interpretability")
        elif model_name == 'Decision Tree':
            obs.append("Captures non-linear patterns, prone to overfitting")
        elif model_name == 'K-Nearest Neighbors':
            obs.append("Instance-based learning, sensitive to feature scaling")
        elif model_name == 'Naive Bayes':
            obs.append("Fast training, assumes feature independence")
        elif model_name == 'Random Forest':
            obs.append("Ensemble method, reduces overfitting")
        elif model_name == 'XGBoost':
            obs.append("Gradient boosting, handles imbalanced data well")
        
        observations[model_name] = ". ".join(obs)
    
    return observations


if __name__ == "__main__":
    print("Loading data...")
    X, y, encoders, df = load_and_preprocess_data('heart.csv')
    print(f"Shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    
    print("\nTraining models...")
    results, models, scaler, test_data = train_and_evaluate_models(X, y)
    
    print("Model Comparison Results")
    
    metrics_df = get_metrics_dataframe(results)
    print(metrics_df.to_string(index=False))
    
    print("Observations")
    
    observations = get_model_observations(results)
    for model, obs in observations.items():
        print(f"\n{model}:")
        print(f"  {obs}")
