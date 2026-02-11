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
    .block-container {
        padding-top: 1rem;
    }
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 1em;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #64B5F6;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00C9FF;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 2px solid #00bcd4;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e0f7fa;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #00838f;
        border: 1px solid #b2ebf2;
        border-bottom: none;
        margin-bottom: -2px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #b2ebf6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff;
        color: #00838f;
        border: 2px solid #00bcd4;
        border-bottom: 2px solid #ffffff;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    .stMetric {
        background: linear-gradient(135deg, #e0f7fa 0%, #b2ebf2 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #00bcd4;
    }
    .stMetric label {
        color: #00838f !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #0f3460 !important;
    }
    .stDataFrame {
        border: 1px solid #00C9FF33;
        border-radius: 10px;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #16213e 100%);
    }
    .stSuccess {
        background: linear-gradient(135deg, #00C9FF22 0%, #92FE9D22 100%);
        border: 1px solid #92FE9D;
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
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost (Ensemble)': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
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
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'],
                annot_kws={'fontsize': 14, 'fontweight': 'bold'},
                linewidths=2, linecolor='white')
    
    for text in ax.texts:
        text.set_color('white' if int(text.get_text()) > cm.max()/2 else '#1a1a2e')
    ax.set_xlabel('Predicted', color='#0f3460')
    ax.set_ylabel('Actual', color='#0f3460')
    ax.set_title(f'Confusion Matrix - {modelName}', color='#00838f', fontweight='bold')
    ax.tick_params(colors='#0f3460')
    return fig

def plotRocCurve(yTrue, yPredProba, modelName):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(yTrue, yPredProba)
    auc = roc_auc_score(yTrue, yPredProba)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#f8fafc')
    ax.set_facecolor('#ffffff')
    ax.plot(fpr, tpr, color='#00838f', lw=3, label=f'ROC curve (AUC = {auc:.4f})')
    ax.fill_between(fpr, tpr, alpha=0.2, color='#00bcd4')
    ax.plot([0, 1], [0, 1], color='#26a69a', linestyle='--', alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', color='#0f3460')
    ax.set_ylabel('True Positive Rate', color='#0f3460')
    ax.set_title(f'ROC Curve - {modelName}', color='#00838f', fontweight='bold')
    ax.legend(loc='lower right', facecolor='#ffffff', edgecolor='#00838f', labelcolor='#0f3460')
    ax.tick_params(colors='#0f3460')
    ax.spines['bottom'].set_color('#b0bec5')
    ax.spines['top'].set_color('#b0bec5')
    ax.spines['left'].set_color('#b0bec5')
    ax.spines['right'].set_color('#b0bec5')
    ax.grid(True, alpha=0.3, color='#b0bec5')
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
    
    modelNames = ['Logistic Regression', 'Decision Tree', 'kNN',
                   'Naive Bayes', 'Random Forest (Ensemble)', 'XGBoost (Ensemble)']
    
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
    st.sidebar.header("Data Upload")
    
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
                     'kNN', 'Naive Bayes', 'Random Forest (Ensemble)', 'XGBoost (Ensemble)']
    selectedModel = st.sidebar.selectbox("Select Model", modelOptions)
    
    # Test size slider
    testSize = st.sidebar.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dataset", "Model Performance", "Visualizations", "Classification Report", "Observations"])
    
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
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#ffffff')
        dataFrame['HeartDisease'].value_counts().sort_index().plot(kind='bar', ax=ax, color=['#00bcd4', '#26a69a'])
        ax.set_xticklabels(['0 (No Disease)', '1 (Disease)'], rotation=0, color='#0f3460')
        ax.set_ylabel('Count', color='#0f3460')
        ax.set_title('Heart Disease Distribution', color='#00838f', fontweight='bold')
        ax.tick_params(colors='#0f3460')
        ax.spines['bottom'].set_color('#b0bec5')
        ax.spines['top'].set_color('#f8fafc')
        ax.spines['left'].set_color('#b0bec5')
        ax.spines['right'].set_color('#f8fafc')
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
            fig.patch.set_facecolor('#f8fafc')
            ax.set_facecolor('#ffffff')
            coolColors = ['#00838f', '#26a69a', '#00bcd4', '#4dd0e1', '#80deea', '#b2ebf2']
            metricsForPlot.plot(kind='bar', ax=ax, width=0.8, color=coolColors)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', color='#0f3460')
            ax.set_ylabel('Score', color='#0f3460')
            ax.set_title('Model Performance Comparison', color='#00838f', fontweight='bold', fontsize=14)
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', facecolor='#ffffff', edgecolor='#00838f', labelcolor='#0f3460')
            ax.set_ylim(0, 1.1)
            ax.tick_params(colors='#0f3460')
            ax.spines['bottom'].set_color('#b0bec5')
            ax.spines['top'].set_color('#f8fafc')
            ax.spines['left'].set_color('#b0bec5')
            ax.spines['right'].set_color('#f8fafc')
            ax.grid(True, alpha=0.3, color='#b0bec5', axis='y')
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
    
    # Tab 5: Observations
    with tab5:
        st.header("Model Performance Observations")
        
        # Generate observations for each model
        observations = {
            'Logistic Regression': {
                'description': 'Linear model that works well for linearly separable data.'
            },
            'Decision Tree': {
                'description': 'Tree-based model that splits data based on feature thresholds.',
            },
            'kNN': {
                'description': 'Instance-based learning that classifies based on nearest neighbors.'
            },
            'Naive Bayes': {
                'description': 'Probabilistic classifier based on Bayes theorem with independence assumption'
            },
            'Random Forest (Ensemble)': {
                'description': 'Ensemble of decision trees using bagging to reduce overfitting.'
            },
            'XGBoost (Ensemble)': {
                'description': 'Gradient boosting algorithm that builds trees sequentially.'
            }
        }
        
        # Create observations table based on actual metrics
        st.subheader("Model Observations Table")
        
        obsData = []
        for modelName, result in allResults.items():
            m = result['metrics']
            acc, auc, mcc = m['Accuracy'], m['AUC'], m['MCC']
            
            # Generate performance observation
            if acc >= 0.85:
                perf = f"Excellent accuracy ({acc:.2%})"
            elif acc >= 0.75:
                perf = f"Good accuracy ({acc:.2%})"
            else:
                perf = f"Moderate accuracy ({acc:.2%})"
            
            if auc >= 0.90:
                perf += ". Outstanding discrimination (AUC: {:.4f})".format(auc)
            elif auc >= 0.80:
                perf += ". Good discrimination (AUC: {:.4f})".format(auc)
            else:
                perf += ". Fair discrimination (AUC: {:.4f})".format(auc)
            
            if mcc >= 0.6:
                perf += ". Strong prediction correlation (MCC: {:.4f})".format(mcc)
            elif mcc >= 0.4:
                perf += ". Moderate prediction correlation (MCC: {:.4f})".format(mcc)
            
            obs = observations.get(modelName, {})
            perf += f". {obs.get('strengths', '')}"
            
            obsData.append({
                'ML Model Name': modelName,
                'Observation about model performance': perf
            })
        
        obsDf = pd.DataFrame(obsData)
        st.dataframe(obsDf, use_container_width=True, hide_index=True)
        
        # Detailed observations for each model
        st.subheader("Detailed Model Analysis")
        
        for modelName, result in allResults.items():
            m = result['metrics']
            obs = observations.get(modelName, {})
            
            with st.expander(f"{modelName}"):
                st.markdown(f"**Description:** {obs.get('description', 'N/A')}")
                st.markdown("**Performance on this dataset:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Accuracy", f"{m['Accuracy']:.4f}")
                    st.metric("AUC", f"{m['AUC']:.4f}")
                with col2:
                    st.metric("Precision", f"{m['Precision']:.4f}")
                    st.metric("Recall", f"{m['Recall']:.4f}")
                with col3:
                    st.metric("F1 Score", f"{m['F1 Score']:.4f}")
                    st.metric("MCC", f"{m['MCC']:.4f}")

if __name__ == "__main__":
    main()
