#!/usr/bin/env python3
"""
Complete Heart Disease Prediction Pipeline
This script runs the entire ML pipeline without using Jupyter notebooks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import dendrogram, linkage
import joblib
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 Starting Complete Heart Disease Prediction Pipeline")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n📊 Step 1: Data Preprocessing")
    print("-" * 30)
    
    # Load the real Heart Disease UCI dataset
    df = pd.read_csv('data/heart_disease_uci.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Data preprocessing
    df_processed = df.copy()
    
    # Handle categorical variables
    df_processed['sex'] = df_processed['sex'].map({'Male': 1, 'Female': 0})
    
    cp_mapping = {
        'typical angina': 0,
        'atypical angina': 1,
        'non-anginal': 2,
        'asymptomatic': 3
    }
    df_processed['cp'] = df_processed['cp'].map(cp_mapping)
    
    df_processed['fbs'] = df_processed['fbs'].fillna(0).astype(int)
    
    restecg_mapping = {
        'normal': 0,
        'lv hypertrophy': 1,
        'st-t abnormality': 2
    }
    df_processed['restecg'] = df_processed['restecg'].map(restecg_mapping)
    
    df_processed['exang'] = df_processed['exang'].fillna(0).astype(int)
    
    slope_mapping = {
        'upsloping': 0,
        'flat': 1,
        'downsloping': 2
    }
    df_processed['slope'] = df_processed['slope'].map(slope_mapping)
    
    thal_mapping = {
        'normal': 1,
        'fixed defect': 2,
        'reversable defect': 3
    }
    df_processed['thal'] = df_processed['thal'].map(thal_mapping)
    
    # Create target variable
    df_processed['target'] = (df_processed['num'] > 0).astype(int)
    
    # Select features
    feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                       'thalch', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    
    df_final = df_processed[feature_columns + ['target']].copy()
    
    # Handle missing values
    for col in df_final.columns:
        if df_final[col].dtype in ['int64', 'float64']:
            df_final[col] = df_final[col].fillna(df_final[col].median())
    
    # Save processed data
    df_final.to_csv('data/heart_disease.csv', index=False)
    
    print(f"✅ Processed dataset shape: {df_final.shape}")
    print(f"Target distribution: {df_final['target'].value_counts().to_dict()}")
    
    # Step 2: Feature Scaling
    print("\n🔧 Step 2: Feature Scaling")
    print("-" * 30)
    
    X = df_final.drop('target', axis=1)
    y = df_final['target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Save scaled data
    X_scaled_df.to_csv('data/X_scaled.csv', index=False)
    y.to_csv('data/y_target.csv', index=False)
    
    print(f"✅ Scaled features shape: {X_scaled_df.shape}")
    
    # Step 3: PCA Analysis
    print("\n📈 Step 3: PCA Analysis")
    print("-" * 30)
    
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Find optimal number of components (90% variance)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    n_components_90 = np.argmax(cumsum >= 0.90) + 1
    
    print(f"✅ Optimal components for 90% variance: {n_components_90}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_[:n_components_90].sum():.3f}")
    
    # Apply PCA with optimal components
    pca_optimal = PCA(n_components=n_components_90)
    X_pca_optimal = pca_optimal.fit_transform(X_scaled)
    X_pca_df = pd.DataFrame(X_pca_optimal, columns=[f'PC{i+1}' for i in range(n_components_90)])
    
    # Save PCA data
    X_pca_df.to_csv('data/X_pca.csv', index=False)
    
    # Step 4: Feature Selection
    print("\n🎯 Step 4: Feature Selection")
    print("-" * 30)
    
    # Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    feature_importance = rf.feature_importances_
    
    # Select top features
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Select top 10 features
    top_features = feature_importance_df.head(10)['feature'].tolist()
    X_selected = X_scaled_df[top_features]
    
    # Save selected features
    X_selected.to_csv('data/X_selected.csv', index=False)
    
    print(f"✅ Selected {len(top_features)} features: {top_features}")
    
    # Step 5: Supervised Learning
    print("\n🤖 Step 5: Supervised Learning")
    print("-" * 30)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Initialize models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'SVM': SVC(random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"✅ {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
    
    # Step 6: Unsupervised Learning
    print("\n🔍 Step 6: Unsupervised Learning")
    print("-" * 30)
    
    # K-Means Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_selected)
    
    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=2)
    hierarchical_labels = hierarchical.fit_predict(X_selected)
    
    # Calculate ARI scores
    kmeans_ari = adjusted_rand_score(y, kmeans_labels)
    hierarchical_ari = adjusted_rand_score(y, hierarchical_labels)
    
    print(f"✅ K-Means ARI: {kmeans_ari:.3f}")
    print(f"✅ Hierarchical ARI: {hierarchical_ari:.3f}")
    
    # Step 7: Hyperparameter Tuning
    print("\n⚙️ Step 7: Hyperparameter Tuning")
    print("-" * 30)
    
    # Tune Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10]
    }
    
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    
    rf_grid.fit(X_train, y_train)
    
    # Get best model
    best_model = rf_grid.best_estimator_
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
    
    best_accuracy = accuracy_score(y_test, y_pred_best)
    best_f1 = f1_score(y_test, y_pred_best)
    
    print(f"✅ Best Random Forest: Accuracy={best_accuracy:.3f}, F1={best_f1:.3f}")
    print(f"Best parameters: {rf_grid.best_params_}")
    
    # Save best model
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(top_features, 'models/selected_features.pkl')
    
    # Step 8: Final Results
    print("\n📊 Step 8: Final Results")
    print("-" * 30)
    
    # Create results summary
    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values('f1', ascending=False)
    
    print("\nModel Performance Summary:")
    print(results_df.round(3))
    
    # Save results
    results_df.to_csv('results/model_performance.csv')
    
    # Create evaluation metrics file
    with open('results/evaluation_metrics.txt', 'w') as f:
        f.write("Heart Disease Prediction - Model Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset Information:\n")
        f.write(f"- Total samples: {len(df_final)}\n")
        f.write(f"- Features: {len(feature_columns)}\n")
        f.write(f"- Target classes: 2 (No Heart Disease, Heart Disease)\n")
        f.write(f"- Train/Test split: 80/20\n\n")
        
        f.write("Best Model Performance:\n")
        f.write(f"- Model: Random Forest (Tuned)\n")
        f.write(f"- Accuracy: {best_accuracy:.3f}\n")
        f.write(f"- F1-Score: {best_f1:.3f}\n")
        f.write(f"- Precision: {precision_score(y_test, y_pred_best):.3f}\n")
        f.write(f"- Recall: {recall_score(y_test, y_pred_best):.3f}\n")
        f.write(f"- AUC: {roc_auc_score(y_test, y_pred_best):.3f}\n\n")
        
        f.write("Selected Features:\n")
        for i, feature in enumerate(top_features, 1):
            f.write(f"{i}. {feature}\n")
    
    print("\n🎉 Pipeline completed successfully!")
    print("Files created:")
    print("- data/heart_disease.csv (processed dataset)")
    print("- data/X_scaled.csv (scaled features)")
    print("- data/X_selected.csv (selected features)")
    print("- data/y_target.csv (target variable)")
    print("- models/best_model.pkl (trained model)")
    print("- models/scaler.pkl (feature scaler)")
    print("- models/selected_features.pkl (feature names)")
    print("- results/model_performance.csv (performance metrics)")
    print("- results/evaluation_metrics.txt (detailed results)")

if __name__ == "__main__":
    main()
