# 🚀 Spaceship Titanic - Kaggle Competition

![Kaggle](https://img.shields.io/badge/Kaggle-Spaceship%20Titanic-blue?logo=kaggle)
![Python](https://img.shields.io/badge/Python-3.9+-yellow?logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-Model-green)

This repository contains my solution to the **[Spaceship Titanic Kaggle competition](https://www.kaggle.com/competitions/spaceship-titanic)**.

**The Challenge:** Predict whether a passenger was **transported to another dimension** after the spaceship accident.

---

## 📂 Repository Structure

```
spaceship-titanic/
├── train.csv              # Training dataset
├── test.csv               # Test dataset  
├── submission.csv         # Final predictions for Kaggle submission
├── notebook.ipynb         # Main notebook (exploration + training)
├── requirements.txt       # Dependencies
└── README.md             # Project documentation
```

---

## 🛠️ Workflow

### 1. Exploratory Data Analysis (EDA)
- Inspected missing values patterns
- Analyzed feature correlations  
- Examined distributions of numerical and categorical features
- Identified key predictive patterns

### 2. Data Preprocessing
- **Missing Value Imputation:** Used `KNNImputer` for numerical features
- **Categorical Encoding:** Applied `pd.get_dummies()` with `drop_first=True`
- **Feature Alignment:** Ensured train/test sets have identical features
- **Data Cleaning:** Removed redundant columns post-encoding

### 3. Model Training & Selection
- **Models Tested:** Logistic Regression, Random Forest, XGBoost, LightGBM
- **Winner:** ✅ **LightGBM (LGBMClassifier)** - Best performance
- **Validation:** Cross-validation for robust model evaluation

### 4. Prediction & Submission
- Generated predictions on test dataset
- Converted outputs to **boolean format** (`True`/`False`) for Kaggle
- Created properly formatted `submission.csv`

---

## 💻 Key Implementation

```python
from lightgbm import LGBMClassifier
from sklearn.impute import KNNImputer
import pandas as pd

# Load datasets
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# Impute missing values
impute_cols = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
imputer = KNNImputer(n_neighbors=5)
df_train[impute_cols] = imputer.fit_transform(df_train[impute_cols])
df_test[impute_cols] = imputer.transform(df_test[impute_cols])

# Encode categorical variables
df_train = pd.get_dummies(df_train, drop_first=True)
df_test = pd.get_dummies(df_test, drop_first=True)

# Align train/test features
df_train, df_test = df_train.align(df_test, join="left", axis=1, fill_value=0)

# Train model
X = df_train.drop(columns=["Transported"])
y = df_train["Transported"]

model = LGBMClassifier(random_state=42)
model.fit(X, y)

# Generate predictions (convert to boolean for Kaggle)
predictions = model.predict(df_test).astype(bool)

# Create submission file
submission = pd.DataFrame({
    "PassengerId": pd.read_csv("test.csv")["PassengerId"],
    "Transported": predictions
})
submission.to_csv("submission.csv", index=False)
```

---

## ✅ Submission Format

The `submission.csv` file must follow this exact format:

```csv
PassengerId,Transported
0013_01,False
0018_01,True
0019_01,False
0021_01,False
...
```

---

## 📦 Installation

### Clone and Setup
```bash
git clone https://github.com/fratsaislam/Spaceship-Stitanic-kaggle.git
cd spaceship-titanic
pip install -r requirements.txt
```

### Run the Analysis
```bash
jupyter notebook notebook.ipynb
```

---

## 🏆 Results

| Metric | Value |
|--------|-------|
| **Best Model** | LightGBM |
| **Validation Accuracy** | 80% |
| **Kaggle Public LB Score** | 0.8 |

---

## 💡 Key Insights

- **LightGBM** efficiently handles both categorical features and missing values
- **KNN Imputation** performed better than simple mean/median imputation
- **Feature Engineering** opportunities exist in passenger groupings and spending patterns
- **Boolean conversion** is crucial for Kaggle submission format

---



## 📝 Dependencies

```txt
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
lightgbm>=3.2.0
jupyter>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## 👨‍💻 Author

**FratsaIslam**
---

<div align="center">
  <i>🌌 Safe travels through the dimensions! 🌌</i>
</div>