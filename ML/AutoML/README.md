# AutoML Evaluation: Wine Quality Classification (H2O AutoML)

**Author:** Diego Armando Villa Domínguez
**Course:** MIACD — AutoML Assignment

## Objective

The goal of this practice is to evaluate the behavior of an **Automated Machine Learning (AutoML)** tool — H2O AutoML — on a classification dataset. Rather than focusing on maximizing model accuracy, the purpose is to understand **which parts of the machine learning workflow get automated** by the tool and **which parts still require human judgment and decision-making**.

## Dataset

**[Wine Quality (red)]([https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv])** from the UCI Machine Learning Repository.

This dataset was chosen because its target variable (`quality`) is **not binary**, which allows the AutoML tool to be evaluated in two different ways:

1. As a **multiclass classification** problem, using the raw `quality` column.
2. As a **binary classification** problem, by engineering a new target that flags high-quality wines.

## Option 1 — Multiclass Classification

Each distinct `quality` score is treated as its own class. Since this is multiclass, **LogLoss** is used as the evaluation metric instead of AUC — it measures how well-calibrated the model's predicted probabilities are, not just whether the predicted class was correct.

```python
# --- HUMAN DECISION: choose the dataset and target column ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/winequality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
objetivo = "quality"

# Start H2O
h2o.init()

# Convert pandas DataFrame to H2OFrame
datos = h2o.H2OFrame(df)

# Predictor variables
x = datos.columns
x.remove(objetivo)

# --- HUMAN DECISION: treat quality as multiclass classification ---
datos[objetivo] = datos[objetivo].asfactor()

# Train/test split
train, test = datos.split_frame(ratios=[0.8], seed=42)

# --- HUMAN DECISION: success metric and time budget ---
aml = H2OAutoML(
    max_runtime_secs=180,
    seed=42,
    sort_metric="logloss"  # Appropriate metric for multiclass classification
)

# Training
aml.train(
    x=x,
    y=objetivo,
    training_frame=train
)

# Leaderboard
print(aml.leaderboard.head())

# Evaluation on the test set
perf = aml.leader.model_performance(test)
print(perf)
```

**Result:** the best model (a Stacked Ensemble) achieved a **LogLoss of 0.7983**, indicating it combined multiple base models effectively for accurate multiclass classification.

## Option 2 — Binary Classification

A new column, `high_q`, is engineered to convert the original 3–8 `quality` scale into a binary label (1 = high quality, 0 = otherwise). For this formulation, **AUC** is the appropriate evaluation metric.

```python
# --- HUMAN DECISION: choose the dataset and target column ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/winequality/winequality-red.csv"
df = pd.read_csv(url, sep=';')

df["high_q"] = (df["quality"] >= 7).astype(int)
objetivo = "high_q"

# Start H2O
h2o.init()

# Convert pandas DataFrame to H2OFrame
datos = h2o.H2OFrame(df)

# Predictor variables
x = datos.columns
x.remove(objetivo)
x.remove("quality")  # avoid data leakage

# --- HUMAN DECISION: treat quality as binary classification ---
datos[objetivo] = datos[objetivo].asfactor()

# Train/test split
train, test = datos.split_frame(ratios=[0.8], seed=42)

# --- HUMAN DECISION: success metric and time budget ---
aml = H2OAutoML(
    max_runtime_secs=180,
    seed=42,
    sort_metric="AUC"  # Appropriate metric for binary classification
)

# Training
aml.train(
    x=x,
    y=objetivo,
    training_frame=train
)

# Leaderboard
print(aml.leaderboard.head())

# Evaluation on the test set
perf = aml.leader.model_performance(test)
print(perf)
```

**Result:** the best model achieved an **AUC of 0.9187**, indicating strong ability to distinguish between high- and low-quality wines.

## What AutoML Automated vs. What Required Human Judgment

| What the tool automated | Option 1: Multiclass | Option 2: Binary |
|---|---|---|
| Tried and compared many algorithms (GBM, XGBoost, Random Forest, GLM, Deep Learning, etc.) and tuned their hyperparameters. | Defining the problem as multiclass classification, choosing the dataset, and setting `quality` as the target variable. | Defining the problem as binary classification, engineering the `high_q` variable (`quality >= 7`), choosing the dataset, and setting `high_q` as the target variable. |
| Performed basic preprocessing (handling categorical variables, missing values, cross-validation, and building ensembles). | Converting `quality` to a factor (`asfactor()`). | Creating `high_q`, removing the original `quality` column to avoid data leakage, and converting `high_q` to a factor (`asfactor()`). |
| Trained all models, compared them, and generated a leaderboard with the relevant metrics. | Choosing LogLoss as the evaluation metric (appropriate for multiclass classification). | Choosing AUC as the evaluation metric (appropriate for binary classification). |
| Computed performance metrics, built the winning model (Stacked Ensemble), and reported variable importance and performance. | Interpreting the leaderboard and LogLoss to judge whether the model is adequate. | Interpreting the leaderboard, AUC, and other metrics to judge whether the model adequately distinguishes high- from low-quality wines. |

## Conclusion

AutoML clearly saves a significant amount of work, but human input remains essential. The analyst must still decide what type of classification (or model) to use, define the correct target variable, and clean the dataset properly.

This was demonstrated firsthand: in an initial attempt at the binary option, the `quality` column was **not removed** from the predictors before training. Since `high_q` was derived directly from `quality`, the model had direct access to the answer — resulting in a (meaningless) AUC of 1.0.

**Takeaway:** AutoML is a powerful tool, but it is only as good as the information it's given — and ensuring that information is correct and leak-free remains the analyst's responsibility.
