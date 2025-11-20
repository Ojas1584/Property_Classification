# Property Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange?logo=xgboost&logoColor=white)
![Optuna](https://img.shields.io/badge/Optuna-HPO-purple?logo=optuna&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Feature%20Engineering-F7931E?logo=scikitlearn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)

A robust text-classification system that predicts property types from address/description text.

##  Table of Contents
- [Problem Statement](#problem-statement)
- [Repository Structure](#repository-structure)
- [Approach](#approach)
- [Results](#results)
- [Setup & Usage](#setup--usage)
- [Model Artifacts](#model-artifacts)

---

##  Problem Statement

Classify properties into one of five categories based on textual descriptions:
- **flat**
- **commercial unit**
- **houseorplot**
- **landparcel**
- **others**

---

##  Repository Structure

```
Property_Classification/
├── best_model/              # Trained model artifacts
│   ├── final_xgb_model.json
│   ├── tfidf_vectorizer.pkl
│   └── label_encoder.pkl
├── results/                 # Evaluation outputs
│   ├── confusion_matrix.png
│   ├── learning_curve_mlogloss.png
│   ├── per_class_f1.png
│   ├── feature_importance_gain_mapped.png
│   ├── feature_importance_weight_mapped.png
│   ├── evals_result.json
│   └── summary.txt
├── data/                    
├── Classification.ipynb     # Main training notebook

```

##  Approach

### 1. **Data Preparation**
- Loaded training and validation datasets
- Applied text normalization: lowercasing, whitespace trimming, special character cleanup
- Stratified sampling to maintain class distribution

### 2. **Feature Engineering**
- **TF-IDF Vectorization** with unigrams and bigrams
- Rationale: Excellent performance on short text fields (property addresses/descriptions)
- Parameters: `max_features=5000, ngram_range=(1,2)`

### 3. **Model Selection**
- **XGBoost Classifier** with `multi:softprob` objective
- Advantages:
  - Handles imbalanced classes well
  - Built-in regularization
  - Efficient training on sparse features

### 4. **Hyperparameter Optimization**
- Framework: **Optuna** (100 trials)
- Optimized parameters:
  - `max_depth`, `learning_rate`, `n_estimators`
  - `min_child_weight`, `subsample`, `colsample_bytree`
- Validation strategy: 5-fold stratified cross-validation

### 5. **Evaluation**
- Metrics: Accuracy, Macro F1, Precision, Recall
- Visualizations: Confusion matrix, learning curves, per-class performance
- Final evaluation on held-out validation set

---

##  Results

### Validation Set Performance
| Metric | Score |
|--------|-------|
| **Accuracy** | 0.91 |
| **Macro F1** | 0.89 |
| **Weighted F1** | 0.91 |

### Per-Class Performance
- **flat**: F1 = 0.92
- **commercial unit**: F1 = 0.88
- **houseorplot**: F1 = 0.90
- **landparcel**: F1 = 0.87
- **others**: F1 = 0.89

The model demonstrates consistent performance across all categories with minimal class bias.

---

##  Setup & Usage

### Prerequisites
```bash
Python 3.10+
```

### Installation
```bash
# Clone the repository
git clone https://github.com/Ojas1584/Property_Classification.git
cd Property_Classification

# Run the notebook
jupyter notebook Classification.ipynb


```

The notebook contains the complete pipeline:
1. Data loading & exploration
2. Preprocessing & feature extraction
3. Model training & hyperparameter tuning
4. Evaluation & visualization
5. Model export

### Using Pre-trained Model
```python
import joblib
import xgboost as xgb

# Load artifacts
model = xgb.XGBClassifier()
model.load_model('best_model/final_xgb_model.json')
vectorizer = joblib.load('best_model/tfidf_vectorizer.pkl')
label_encoder = joblib.load('best_model/label_encoder.pkl')

# Predict
text = ["3 bedroom flat in downtown"]
X = vectorizer.transform(text)
prediction = model.predict(X)
category = label_encoder.inverse_transform(prediction)
print(category)
```

---

## Model Artifacts

All components required for reproducibility are saved in `best_model/`:

| File | Description |
|------|-------------|
| `final_xgb_model.json` | Trained XGBoost model |
| `tfidf_vectorizer.pkl` | Fitted TF-IDF vectorizer |
| `label_encoder.pkl` | Label encoder for class mapping |

These ensure end-to-end reproducibility without requiring retraining.

---

##  Technologies Used
- **Python 3.10+**
- **XGBoost**: Gradient boosting classifier
- **Scikit-learn**: Feature engineering, metrics
- **Optuna**: Hyperparameter optimization
- **Pandas**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

---

##  Notes
- Model training takes ~10-15 minutes on standard hardware

