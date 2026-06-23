# Systematic Model Comparison — Telco Customer Churn

Course assignment for **Machine Learning with Python, scikit-learn and Keras**
(DHBW Stuttgart, Summer Semester 2026).

The goal is a complete, **methodically sound and critically reflected** ML pipeline that
systematically compares **five models** on the [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
dataset. The focus is *not* maximum accuracy, but well-justified decisions — every choice is
documented directly in the notebook with Markdown cells.

## Task

Binary classification: **does a customer churn?** (`Churn` = Yes / No)
- ~7,000 customers, 20 features (numeric and categorical)
- Slight class imbalance (~27 % churn) → accuracy alone is misleading; F1 and ROC-AUC matter
- Categorical features must be encoded; `TotalCharges` has 11 hidden missing values (new customers, `tenure = 0`)

### Models compared

| # | Model | Type |
|---|-------|------|
| 1 | Logistic Regression | classic |
| 2 | Decision Tree / Random Forest / XGBoost *(pick one)* | classic |
| 3 | k-NN / SVM *(pick one)* | classic |
| 4 | MLP Variant A — `Dense(64) → Dense(1)` | neural |
| 5 | MLP Variant B — `Dense(64) → Dropout(0.3) → Dense(32) → Dense(1)` | neural |

## Repository structure

```
.
├── Assignment.ipynb          # ← main deliverable (the notebook you work in)
├── README.md
├── data/
│   └── telco_customer_churn.csv
├── docs/
│   ├── Assignment_Brief.pdf            # original task description
│   ├── Management_Summary.md           # deliverable: results summary
│   └── ML_Sommersemester_Schneider.pdf # full course script
└── course_materials/         # reference notebooks from the lectures
    ├── student/              # student versions (exercises)
    └── instructor/           # instructor versions (solutions)
```

## Notebook workflow (7 steps)

1. **Data analysis** — shapes, dtypes, missing values, target distribution, ≥3 visualizations
2. **Preprocessing** — handle missing values, encode categoricals, stratified 70/15/15 split, scaling
3. **Classic models** — train 3 models, 5-fold stratified CV, report Accuracy / F1 / ROC-AUC
4. **Neural net Variant A** — Adam, early stopping, learning curves
5. **Neural net Variant B** — SGD + dropout, compare against A
6. **Comparison, tuning & final evaluation** — GridSearchCV / KerasTuner, **test set touched exactly once**, confusion matrix
7. **Reflection** — production recommendation, limits, surprises

> Step 1 is fully implemented. Steps 2–7 contain a scaffold with the concrete sub-tasks to complete.

## How to run

The notebook is designed for **Google Colab** (recommended: GPU runtime for the Keras models).

The data-loading cell tries three sources in order:
1. local file `data/telco_customer_churn.csv`
2. automatic download via `kagglehub` (`blastchar/telco-customer-churn`) — recommended on Colab
3. manual upload dialog (Colab fallback)

For a local run you need: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `tensorflow`,
`xgboost` (optional), `kagglehub` (optional).

## Deliverables

1. Fully executed `Assignment.ipynb` (all cells with Markdown commentary)
2. PDF export of the notebook
3. Management summary ([docs/Management_Summary.md](docs/Management_Summary.md))

## Note on AI usage

AI tools are explicitly permitted. AI-generated code blocks are marked with `# Quelle: <tool>`.
Interpretations, justifications and the final recommendation are written by the group, not the AI.
