import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Heart Disease Prediction - ML Assignment",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #6C3483;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #5D6D7E;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F4ECF7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #8E44AD;
    }
</style>
""", unsafe_allow_html=True)

def fetchDefaultHeartData():
    """Load the default heart disease dataset"""
    try:
        dataFrame = pd.read_csv('heart.csv')
        return dataFrame
    except:
        return None

def preprocessData(dataFrame, targetCol='HeartDisease'):
    """Preprocess the dataset"""
    dfProcessed = dataFrame.copy()
    
    # Identify categorical columns
    categoricalCols = dfProcessed.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables
    labelEncoders = {}
    for col in categoricalCols:
        le = LabelEncoder()
        dfProcessed[col] = le.fit_transform(dfProcessed[col].astype(str))
        labelEncoders[col] = le
    
    # Separate features and target
    if targetCol in dfProcessed.columns:
        X = dfProcessed.drop(targetCol, axis=1)
        y = dfProcessed[targetCol]
    else:
        X = dfProcessed
        y = None
    
    return X, y, labelEncoders

def getModel(modelName):
    """Return the selected model"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    return models.get(modelName)

def calculateAllMetrics(yTrue, yPred, yPredProba=None):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(yTrue, yPred),
        'Precision': precision_score(yTrue, yPred, average='weighted', zero_division=0),
        'Recall': recall_score(yTrue, yPred, average='weighted', zero_division=0),
        'F1 Score': f1_score(yTrue, yPred, average='weighted', zero_division=0),
        'MCC': matthews_corrcoef(yTrue, yPred)
    }
    
    if yPredProba is not None:
        try:
            metrics['AUC'] = roc_auc_score(yTrue, yPredProba)
        except:
            metrics['AUC'] = 0.0
    else:
        metrics['AUC'] = 0.0
    
    return metrics

def plotConfusionMatrix(cm, modelName):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix - {modelName}')
    return fig

def plotRocCurve(yTrue, yPredProba, modelName):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(yTrue, yPredProba)
    auc = roc_auc_score(yTrue, yPredProba)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#8E44AD', lw=2, label=f'ROC curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {modelName}')
    ax.legend(loc='lower right')
    return fig

@st.cache_data
def trainAllModels(_X, _y, testSize=0.2):
    """Train all models and return results based on the selected test size"""
    print(f"->>> Training models with test size: {testSize}")
    
    xTrain, xTest, yTrain, yTest = train_test_split(
        _X, _y, test_size=testSize, random_state=42, stratify=_y
    )
    
    scaler = StandardScaler()
    xTrainScaled = scaler.fit_transform(xTrain)
    xTestScaled = scaler.transform(xTest)
    
    modelNames = ['Logistic Regression', 'Decision Tree', 'K-Nearest Neighbors',
                   'Naive Bayes', 'Random Forest', 'XGBoost']
    
    allResults = {}
    
    for name in modelNames:
        model = getModel(name)
        model.fit(xTrainScaled, yTrain)
        
        yPred = model.predict(xTestScaled)
        yPredProba = None
        if hasattr(model, 'predict_proba'):
            yPredProba = model.predict_proba(xTestScaled)[:, 1]
        
        metrics = calculateAllMetrics(yTest, yPred, yPredProba)
        cm = confusion_matrix(yTest, yPred)
        report = classification_report(yTest, yPred, output_dict=True)
        
        allResults[name] = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'confusionMatrix': cm,
            'classificationReport': report,
            'yTest': yTest,
            'yPred': yPred,
            'yPredProba': yPredProba
        }
    
    return allResults

def main():
    # Header
    st.markdown('<h1 class="main-header">ML Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Assignment 2 ( Prasad Kadu 2025AA05195 ) </p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("📁 Data Upload")
    
    # File upload option
    uploadedFile = st.sidebar.file_uploader(
        "Upload CSV dataset (test data)",
        type=['csv'],
        help="Upload your own CSV file for testing"
    )
    
    # Load data
    if uploadedFile is not None:
        dataFrame = pd.read_csv(uploadedFile)
        st.sidebar.success("Custom dataset loaded!")
    else:
        dataFrame = fetchDefaultHeartData()
        if dataFrame is not None:
            st.sidebar.info("Using default Heart Disease dataset")
        else:
            st.error("No dataset available. Please upload a CSV file.")
            return
    
    # Dataset info
    st.sidebar.markdown("---")
    st.sidebar.header("Dataset Info")
    st.sidebar.write(f"**Rows:** {dataFrame.shape[0]}")
    st.sidebar.write(f"**Columns:** {dataFrame.shape[1]}")
    
    # Model selection
    st.sidebar.markdown("---")
    st.sidebar.header("Model Selection")
    
    modelOptions = ['All Models', 'Logistic Regression', 'Decision Tree', 
                     'K-Nearest Neighbors', 'Naive Bayes', 'Random Forest', 'XGBoost']
    selectedModel = st.sidebar.selectbox("Select Model", modelOptions)
    
    # Test size slider
    testSize = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Model Performance", "Visualizations", "Classification Report"])
    
    # Preprocess data
    X, y, encoders = preprocessData(dataFrame)
    
    if y is None:
        st.error("Target column 'HeartDisease' not found in dataset!")
        return
    
    # Train all models
    with st.spinner("Training models..."):
        st.toast(f"Test size: {testSize}")  # shows popup in UI
        allResults = trainAllModels(X.values, y.values, testSize)
    
    # Tab 1: Dataset Overview
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("First 10 Rows")
            st.dataframe(dataFrame.head(10), use_container_width=True)
        
        with col2:
            st.subheader("Dataset Statistics")
            st.dataframe(dataFrame.describe(), use_container_width=True)
        
        st.subheader("Target Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        dataFrame['HeartDisease'].value_counts().plot(kind='bar', ax=ax, color=['#1ABC9C', '#E74C3C'])
        ax.set_xticklabels(['No Disease', 'Disease'], rotation=0)
        ax.set_ylabel('Count')
        ax.set_title('Heart Disease Distribution')
        st.pyplot(fig)
    
    # Tab 2: Model Performance
    with tab2:
        st.header("Model Performance Metrics")
        
        if selectedModel == 'All Models':
            # Show comparison table
            st.subheader("Model Comparison Table")
            
            metricsData = []
            for modelName, result in allResults.items():
                row = {'Model': modelName}
                row.update(result['metrics'])
                metricsData.append(row)
            
            metricsDf = pd.DataFrame(metricsData)
            cols = ['Model', 'Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
            metricsDf = metricsDf[cols]
            
            # Format as percentages for display
            for col in cols[1:]:
                metricsDf[col] = metricsDf[col].apply(lambda x: f"{x:.4f}")
            
            st.dataframe(metricsDf, use_container_width=True, hide_index=True)
            
            # Best model highlight
            bestModel = max(allResults.items(), key=lambda x: x[1]['metrics']['Accuracy'])
            st.success(f"**Best Model (by Accuracy):** {bestModel[0]} - {bestModel[1]['metrics']['Accuracy']:.4f}")
            
            # Metrics comparison chart
            st.subheader("Metrics Comparison")
            metricsForPlot = pd.DataFrame(metricsData)
            metricsForPlot = metricsForPlot.set_index('Model')
            
            fig, ax = plt.subplots(figsize=(12, 6))
            metricsForPlot.plot(kind='bar', ax=ax, width=0.8)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_ylabel('Score')
            ax.set_title('Model Performance Comparison')
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
            ax.set_ylim(0, 1.1)
            plt.tight_layout()
            st.pyplot(fig)
            
        else:
            # Show single model metrics
            result = allResults[selectedModel]
            metrics = result['metrics']
            
            st.subheader(f"{selectedModel} Metrics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
                st.metric("Precision", f"{metrics['Precision']:.4f}")
            with col2:
                st.metric("AUC Score", f"{metrics['AUC']:.4f}")
                st.metric("Recall", f"{metrics['Recall']:.4f}")
            with col3:
                st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
                st.metric("MCC", f"{metrics['MCC']:.4f}")
    
    # Tab 3: Visualizations
    with tab3:
        st.header("Model Visualizations")
        
        if selectedModel == 'All Models':
            # Show confusion matrices for all models
            st.subheader("Confusion Matrices")
            
            cols = st.columns(3)
            for idx, (modelName, result) in enumerate(allResults.items()):
                with cols[idx % 3]:
                    fig = plotConfusionMatrix(result['confusionMatrix'], modelName)
                    st.pyplot(fig)
                    plt.close()
        else:
            result = allResults[selectedModel]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Confusion Matrix")
                fig = plotConfusionMatrix(result['confusionMatrix'], selectedModel)
                st.pyplot(fig)
                plt.close()
            
            with col2:
                if result['yPredProba'] is not None:
                    st.subheader("ROC Curve")
                    fig = plotRocCurve(result['yTest'], result['yPredProba'], selectedModel)
                    st.pyplot(fig)
                    plt.close()
    
    # Tab 4: Classification Report
    with tab4:
        st.header("Classification Report")
        
        if selectedModel == 'All Models':
            modelForReport = st.selectbox("Select model for detailed report", 
                                            list(allResults.keys()))
            result = allResults[modelForReport]
        else:
            result = allResults[selectedModel]
            modelForReport = selectedModel
        
        st.subheader(f"{modelForReport} - Detailed Classification Report")
        
        report = result['classificationReport']
        reportDf = pd.DataFrame(report).transpose()
        reportDf = reportDf.round(4)
        st.dataframe(reportDf, use_container_width=True)
        
        # Confusion matrix values
        confusionMatrix = result['confusionMatrix']
        st.subheader("Confusion Matrix Values")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("True Negatives", confusionMatrix[0][0])
        with col2:
            st.metric("False Positives", confusionMatrix[0][1])
        with col3:
            st.metric("False Negatives", confusionMatrix[1][0])
        with col4:
            st.metric("True Positives", confusionMatrix[1][1])

if __name__ == "__main__":
    main()
