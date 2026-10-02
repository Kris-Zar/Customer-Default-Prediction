# 🏦 Customer Loan Default Prediction

A machine learning project that predicts whether a loan applicant will default, using an **AdaBoost Classifier** trained on historical loan data. Designed to help financial institutions make smarter, data-driven lending decisions.

---

## 📌 Repository Description

> Predicts customer loan default risk using AdaBoost on a structured loan dataset — covering end-to-end preprocessing, encoding, model training, and evaluation with a confusion matrix heatmap.

---

## 📁 Project Structure

```
customer-default-prediction/
│
├── Customer_Default_Prediction.ipynb   # Main Jupyter Notebook (EDA + Model)
├── LoanDataset.csv                     # Input dataset (loan records)
└── README.md                           # Project documentation
```

---

## 📊 Dataset

The dataset (`LoanDataset.csv`) contains historical loan records with the following types of features:

| Feature | Description |
|---|---|
| `customer_id` | Unique identifier for each customer |
| `customer_income` | Annual income of the customer |
| `loan_amnt` | Requested loan amount |
| `home_ownership` | Type of home ownership (e.g., RENT, OWN, MORTGAGE) |
| `loan_intent` | Purpose of the loan (e.g., EDUCATION, MEDICAL) |
| `loan_grade` | Credit grade assigned to the loan |
| `historical_default` | Whether the customer has defaulted before |
| `Current_loan_status` | **Target variable** — DEFAULT or NO DEFAULT |

---

## ⚙️ Workflow

### 1. Data Loading & Exploration
- Dataset loaded using `pandas`
- Descriptive statistics generated with `.describe()`

### 2. Data Preprocessing
- **Numerical columns:** Missing values filled with column **mean**
- **Categorical columns:** Missing values filled with column **mode**
- **String cleaning:** Commas removed from `customer_income` and `loan_amnt`, then cast to numeric

### 3. Feature Encoding
Categorical features encoded using `LabelEncoder`:
- `home_ownership`
- `loan_intent`
- `loan_grade`
- `historical_default`
- `Current_loan_status` (target)

### 4. Train-Test Split
- 80% training / 20% testing
- `random_state=42` for reproducibility
- Additional `SimpleImputer` (mean strategy) applied post-split to prevent data leakage

### 5. Model Training
- **Algorithm:** `AdaBoostClassifier`
- **Estimators:** 100 weak learners
- **Random State:** 42

### 6. Evaluation
- **Metric:** Accuracy Score
- **Visualization:** Confusion Matrix heatmap (Seaborn, Green palette)
  - Classes: `Default` vs `No Default`

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | ML pipeline (split, impute, encode, model, metrics) |

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

```bash
jupyter notebook Customer_Default_Prediction.ipynb
```

> **Note:** Update the dataset path in the notebook from the hardcoded local path to your own:
> ```python
> # Change this:
> data = pd.read_csv("C:\\Study content\\ML\\DATASETS\\LoanDataset.csv")
>
> # To this:
> data = pd.read_csv("LoanDataset.csv")
> ```

---

## 📈 Results

The model outputs:
- **Accuracy Score** on the test set
- **Confusion Matrix** visualized as a heatmap showing true vs. predicted loan default labels

---

## 🔮 Possible Improvements

- Try other ensemble methods (Random Forest, XGBoost, Gradient Boosting) for comparison
- Perform hyperparameter tuning with `GridSearchCV`
- Add feature importance visualization
- Address class imbalance using SMOTE or class weights
- Use cross-validation instead of a single train-test split
- Replace `LabelEncoder` with `OrdinalEncoder` or one-hot encoding for nominal features

---

## 📄 License

This project is intended for educational and learning purposes.
