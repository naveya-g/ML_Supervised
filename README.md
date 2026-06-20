# 🏥🏦 ML Supervised Learning - Healthcare Diagnosis & Bank Loan Prediction

This project applies supervised machine learning classification algorithms to solve two real-world problems:
predicting patient medical conditions from biomechanics data, and identifying potential bank customers likely to take a loan on their credit card.

---

### Project Objective

To build machine learning models that can learn from historical data and make accurate predictions - 
(I) Help healthcare teams identify patient conditions early 
(II) Help banks run smarter, targeted marketing campaigns without increasing their budget.

---

### Dataset Summary

**Part 1 - Healthcare (Medical Biomechanics)**
- Patient records from Medical Research University X, provided across 3 separate CSV files (Normal, Type_H, Type_S) masked for confidentiality
- Key features: P_incidence, P_tilt, L_angle, S_slope, P_radius, S_Degree (6 biomechanics attributes)
- Target variable: **Patient Condition Class** - Normal, Type_H, Type_S
- Combined dataset: **310 patient records, 7 features**
- Key challenges: inconsistent class labels across files (e.g. `Nrmal`, `type_h`, `tp_s`), right-skewed distribution in S_Degree

---
**Part 2 - Banking & Marketing**
- Customer data from Bank X split across 2 CSV files, merged on Customer ID
- Key features: Age, HighestSpend, HiddenScore, MonthlyAverageSpend, Level, Mortgage, Security, FixedDepositAccount, InternetBanking, CreditCard
- Target variable: **LoanOnCard** - whether a customer will convert to a loan on credit card (binary: 0 / 1)
- Key challenges: severe class imbalance (4000+ non-loan vs <1000 loan customers), 0.4% missing values in target column, mixed data types in categorical features

---
### Analysis Workflow
 
**Part 1 — Healthcare**
- Fixed inconsistent class label variations across all 3 files (`Nrmal → Normal`, `type_h → Type_H`, `tp_s → Type_S`) and merged them into a single dataset of 310 records using `pd.concat()`
- Visualized a **correlation heatmap** — found strongest correlation between P_incidence and S_slope (0.81); P_radius showed weak/negative correlation with other features
- Plotted **pairplot and jointplot** — S_Degree had the highest influence on class separation; Type_S showed the most spread across all features
- Applied **StandardScaler** for normalization (required for KNN as it is distance-based) and performed 80:20 train-test split
---
**Part 2 — Banking & Marketing**
- Merged `Data1` and `Data2` on `ID`; dropped non-informative columns (`ID`, `ZipCode`, `CustomerSince`) and handled 0.4% null values in the target column
- Converted binary and categorical columns to appropriate types; addressed severe class imbalance (4000+ non-loan vs <1000 loan customers) using **SMOTE**
- Applied `SimpleImputer` for post-SMOTE missing value handling and `StandardScaler` for feature normalization before model training
- Evaluated and compared 3 models — Logistic Regression (with and without SMOTE), SVM, and KNN — on a 75:25 train-test split

---

### Model Performance

**Part 1 - Healthcare (KNN Classifier)**

| Model | Parameters | Test Accuracy |
|---|---|---|
| Baseline KNN | n_neighbors=5, metric=Minkowski | ~84% |
| Optimized KNN | n_neighbors=5, weights=distance, p=2 | **85.48%** |

> Best K values: 5, 9, 11, and 13 all achieved 85.48% with `weights=distance`. K=5 selected as optimal for balance between accuracy and simplicity.

**Part 2 - Banking & Marketing (Multi-Model Comparison)**

| Model | Key Setting | Test Accuracy |
|---|---|---|
| Logistic Regression (without SMOTE) | Default | Lower recall on minority class |
| Logistic Regression (with SMOTE) | Balanced training data | Improved minority class prediction |
| SVM | kernel=RBF | High accuracy |
| KNN | n_neighbors=5 | **~96%** |

---

### Tools Used

- Python, NumPy, Pandas, Matplotlib, Seaborn
- ML Models: `KNeighborsClassifier`, `LogisticRegression`, `SVC` (Scikit-learn)
- Imbalanced Data: `SMOTE` (imbalanced-learn)
- Preprocessing: `StandardScaler`, `SimpleImputer`, `train_test_split`
- Evaluation: Accuracy Score, Classification Report (Precision, Recall, F1), Confusion Matrix
- Environment: Google Colab

---

### 🔍 Use Case

**Part 1** assists medical researchers in automatically classifying patient conditions from biomechanics test results - reducing dependency on manual diagnosis and speeding up clinical decisions in confidential research settings.

**Part 2** enables the bank's marketing team to focus their campaigns only on customers who are most likely to take a loan on their credit card - increasing conversion rates from single-digit to double-digit without increasing the campaign budget.
