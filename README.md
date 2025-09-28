# ❤️ Heart Disease Prediction System
A comprehensive machine learning pipeline for predicting heart disease risk using the UCI Heart Disease Dataset. This project implements advanced data preprocessing, feature selection, dimensionality reduction, supervised and unsupervised learning, hyperparameter tuning, and deployment with a modern Streamlit web interface.

## 📊 Project Overview

This project demonstrates a complete end-to-end machine learning workflow for medical diagnosis prediction, featuring:

- **Advanced Data Preprocessing**: Missing value imputation, categorical encoding, feature scaling
- **Dimensionality Reduction**: Principal Component Analysis (PCA) for feature optimization
- **Intelligent Feature Selection**: Multiple algorithms for optimal feature subset selection
- **Supervised Learning**: Multiple classification algorithms with performance comparison
- **Unsupervised Learning**: Clustering analysis for pattern discovery
- **Hyperparameter Optimization**: Automated model tuning for best performance
- **Interactive Web Interface**: Modern Streamlit UI for real-time predictions
- **Production Deployment**: Ready for cloud deployment with Ngrok

## 🎯 Key Features

### 🔬 **Machine Learning Pipeline**
- **Data Preprocessing**: Handles missing values, categorical encoding, and feature scaling
- **PCA Analysis**: Dimensionality reduction while retaining 90% variance
- **Feature Selection**: Random Forest importance, RFE, and statistical tests
- **Model Training**: Logistic Regression, Decision Trees, Random Forest, SVM
- **Clustering**: K-Means and Hierarchical clustering for pattern discovery
- **Hyperparameter Tuning**: GridSearchCV and RandomizedSearchCV optimization

### 🚀 **Performance Metrics**
- **Best Model**: Tuned Random Forest Classifier
- **Accuracy**: 83.2%
- **F1-Score**: 85.4%
- **Precision**: 84.1%
- **Recall**: 86.7%
- **AUC**: 90.1%

### 💻 **Web Application**
- **Interactive Interface**: User-friendly form for patient data input
- **Real-time Predictions**: Instant heart disease risk assessment
- **Visual Feedback**: Clear risk indicators and recommendations
- **Responsive Design**: Modern, professional UI

## 🏗️ Project Structure

```
Heart_Disease_Project/
├── 📁 data/                          # Dataset and processed data
│   ├── heart_disease_uci.csv         # Original UCI dataset
│   ├── X_scaled.csv                  # Scaled features
│   ├── X_selected.csv                # Selected features
│   ├── X_pca.csv                     # PCA-transformed data
│   └── y_target.csv                  # Target variable
├── 📁 notebooks/                     # Jupyter notebooks for analysis
│   ├── 01_data_preprocessing.ipynb  # Data cleaning and preprocessing
│   ├── 02_pca_analysis.ipynb        # Dimensionality reduction
│   ├── 03_feature_selection.ipynb    # Feature selection algorithms
│   ├── 04_supervised_learning.ipynb  # Classification models
│   ├── 05_unsupervised_learning.ipynb # Clustering analysis
│   └── 06_hyperparameter_tuning.ipynb # Model optimization
├── 📁 models/                        # Trained models and artifacts
│   ├── best_model.pkl               # Best performing model
│   ├── scaler.pkl                   # Feature scaler
│   └── selected_features.pkl        # Selected feature names
├── 📁 ui/                           # Web application
│   └── app.py                       # Streamlit application
├── 📁 deployment/                   # Deployment configurations
│   └── ngrok_setup.txt             # Ngrok deployment guide
├── 📁 results/                      # Analysis results
│   ├── evaluation_metrics.txt       # Model performance metrics
│   └── model_performance.csv        # Detailed performance comparison
├── run_complete_pipeline.py         # End-to-end pipeline execution
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (tested with Python 3.13)
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/mido-io/Heart_Disease_Project.git
   cd Heart_Disease_Project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**
   ```bash
   python run_complete_pipeline.py
   ```

4. **Launch the web application**
   ```bash
   streamlit run ui/app.py
   ```

5. **Access the application**
   - Local: http://localhost:8501
   - Network: http://192.168.0.102:8501

## 📈 Dataset Information

### **UCI Heart Disease Dataset**
- **Total Samples**: 920 patients
- **Features**: 13 clinical attributes
- **Target**: Binary classification (No Heart Disease / Heart Disease)
- **Missing Values**: Handled with intelligent imputation
- **Data Quality**: High-quality medical dataset with comprehensive preprocessing

### **Feature Description**
| Feature | Description | Type |
|---------|-------------|------|
| age | Age in years | Numerical |
| sex | Gender (Male/Female) | Categorical |
| cp | Chest pain type | Categorical |
| trestbps | Resting blood pressure | Numerical |
| chol | Serum cholesterol | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | ST slope | Categorical |
| ca | Number of major vessels | Numerical |
| thal | Thalassemia | Categorical |

## 🔬 Technical Implementation

### **Data Preprocessing Pipeline**
1. **Missing Value Handling**: Intelligent imputation using median and mode
2. **Categorical Encoding**: Label encoding for ordinal variables
3. **Feature Scaling**: StandardScaler for normalization
4. **Data Validation**: Comprehensive quality checks

### **Feature Selection Methods**
- **Random Forest Importance**: Tree-based feature ranking
- **Recursive Feature Elimination**: Backward selection with cross-validation
- **Statistical Tests**: F-test for feature significance
- **Final Selection**: Top 10 most important features

### **Model Architecture**
- **Logistic Regression**: Linear baseline model
- **Decision Tree**: Interpretable tree-based model
- **Random Forest**: Ensemble method with 200 trees
- **Support Vector Machine**: Kernel-based classification
- **Best Model**: Tuned Random Forest (GridSearchCV optimized)

### **Evaluation Metrics**
- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

## 🎨 Web Application Features

### **User Interface**
- **Modern Design**: Clean, professional medical interface
- **Responsive Layout**: Works on desktop and mobile devices
- **Interactive Forms**: User-friendly input validation
- **Real-time Feedback**: Instant prediction results

### **Prediction Features**
- **Risk Assessment**: Binary heart disease prediction
- **Probability Scores**: Confidence levels for predictions
- **Medical Recommendations**: Professional health advice
- **Visual Indicators**: Color-coded risk levels

### **Data Visualization**
- **Performance Metrics**: Model accuracy and performance charts
- **Feature Importance**: Visual representation of key factors
- **Distribution Analysis**: Data exploration and insights

## 🚀 Deployment Options

### **Local Deployment**
```bash
streamlit run ui/app.py
```

### **Ngrok Deployment** (Public Access)
1. Install Ngrok: https://ngrok.com/download
2. Run the application: `streamlit run ui/app.py`
3. In another terminal: `ngrok http 8501`
4. Share the public URL for remote access

### **Cloud Deployment**
- **Heroku**: Container-based deployment
- **AWS**: EC2 instance with load balancing
- **Google Cloud**: App Engine or Compute Engine
- **Azure**: App Service or Container Instances

## 📊 Performance Analysis

### **Model Comparison**
| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| Logistic Regression | 81.5% | 82.1% | 83.2% | 82.6% | 88.3% |
| Decision Tree | 78.3% | 79.1% | 80.5% | 79.8% | 85.7% |
| Random Forest | 82.1% | 83.4% | 84.2% | 83.8% | 89.2% |
| SVM | 80.7% | 81.8% | 82.9% | 82.3% | 87.6% |
| **Random Forest (Tuned)** | **83.2%** | **84.1%** | **86.7%** | **85.4%** | **90.1%** |

### **Feature Importance**
1. **Chest Pain Type (cp)**: 18.5% importance
2. **Serum Cholesterol (chol)**: 15.2% importance
3. **Maximum Heart Rate (thalch)**: 14.8% importance
4. **Age**: 12.3% importance
5. **ST Depression (oldpeak)**: 11.7% importance

## 🔧 Advanced Configuration

### **Environment Variables**
```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### **Model Retraining**
```bash
python run_complete_pipeline.py --retrain
```

### **Custom Feature Selection**
Modify `run_complete_pipeline.py` to adjust feature selection parameters:
```python
# Select top N features
selected_features_names = feature_importance_df.head(N)['feature'].tolist()
```

## 📚 Dependencies

### **Core Libraries**
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **streamlit**: Web application framework
- **joblib**: Model serialization

### **Version Compatibility**
- Python 3.8+ (tested with 3.13)
- scikit-learn 1.3+
- pandas 2.0+
- streamlit 1.28+

## 🤝 Contributing

### **Development Setup**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Install development dependencies: `pip install -r requirements.txt`
4. Make your changes and test thoroughly
5. Submit a pull request

### **Code Standards**
- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation for API changes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Heart Disease dataset
- **Scikit-learn** team for the excellent ML library
- **Streamlit** team for the intuitive web framework
- **Medical professionals** who provided domain expertise

## 📞 Support

For questions, issues, or contributions:
- **Issues**: Create a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Email**: [abdelhamidfarhat@outlook.com]

## 🔮 Future Enhancements

- **Deep Learning Models**: Neural network implementation
- **Real-time Data**: Live patient monitoring integration
- **Mobile App**: React Native or Flutter application
- **API Development**: RESTful API for third-party integration
- **Advanced Analytics**: Time-series analysis and trend prediction
- **Multi-language Support**: Internationalization features

---

**⚠️ Medical Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

**🔒 Privacy Notice**: All patient data is processed locally. No personal information is transmitted to external servers.

---

*Built with ❤️ for advancing healthcare through machine learning*