# Rule-Based Classification: San Francisco vs New York Housing

**Author:** Diego Armando Villa Domínguez
**Course:** MIACD — Rule Learning Assignment

## Overview

This project classifies real estate listings as belonging to **San Francisco (SF)** or **New York (NY)** using hand-crafted and iteratively-optimized classification rules — no machine learning models, just threshold-based logic derived from exploratory data analysis (EDA).

The dataset contains housing listings with the following features:

| Column | Description |
|---|---|
| `beds` | Number of bedrooms |
| `bath` | Number of bathrooms |
| `price` | Listing price |
| `year_built` | Year the property was built |
| `sqft` | Total square footage |
| `price_per_sqft` | Price divided by square footage |
| `elevation` | Elevation of the property |
| `tag` / `in_sf` | Ground-truth city label (NY or SF) |

## Tools

- **Language:** Python
- **Editor:** Visual Studio Code

## Methodology

### 1. Initial EDA

A pairwise scatter plot matrix was generated across all numeric variables to identify which feature pairs best separate the two classes. `elevation` vs `price_per_sqft` showed the clearest visual separation between NY and SF listings.

### 2. Manual Rule Method

Based on visual inspection of the `elevation` vs `price_per_sqft` scatter plot, the following manual rules were defined:

```
Elevation < 40        -> NY
Price per sqft < 1000 -> SF
```

**Result:** 80.49% accuracy

### 3. Iterative Threshold Search

To move beyond manual guesswork, a brute-force grid search was implemented, testing every combination of `price_per_sqft` and `elevation` thresholds present in the dataset:

```python
best_acc = 0      # Best accuracy found so far
best_pps = None   # price_per_sqft threshold that produced it
best_ele = None   # elevation threshold that produced it

for pps in sorted(df["price_per_sqft"].unique()):
    for ele in sorted(df["elevation"].unique()):
        pred = ((df["price_per_sqft"] > pps) & (df["elevation"] < ele)).map(
            {True: "NY", False: "SF"}
        )
        acc = (pred == df["tag"]).mean()

        if acc > best_acc:
            best_acc = acc
            best_pps = pps
            best_ele = ele

print(best_pps, best_ele, best_acc)
```

**Rule of thumb used:** lower elevation + higher price per sqft → NY; otherwise → SF.

**Best result:** `price_per_sqft = 804`, `elevation = 29` → **84.34% accuracy**

### 4. Exploring Additional Variables

Group-by mean and median statistics (by city) were computed for all features to identify which additional variables showed the largest separation between classes. Aside from `elevation` and `price_per_sqft`, `price` and `sqft` showed the most promise.

- Optimizing `price` and `sqft` alone: **65.45% accuracy** (`price = 998000`, `sqft = 1440`)
- Combining all four rules (`elevation`, `price_per_sqft`, `price`, `sqft`) together: **82.52% accuracy** — actually *lower* than the two-variable result, showing that adding more rules doesn't always help.

### 5. Three-Variable Nested Search

A three-way nested search was run, always keeping `elevation` (identified as the highest-impact variable) plus one of the two remaining candidates:

- `elevation`, `price_per_sqft`, `price`
- `elevation`, `price_per_sqft`, `sqft`
- `elevation`, `price`, `sqft`

The best-performing combination was **`elevation` + `price_per_sqft` + `sqft`**:

```
elevation       = 34
price_per_sqft  = 1053
sqft            = 756
```

**Result: 87.40% accuracy**

## Final Evaluation

A confusion matrix was generated for the best rule set:

| | Predicted NY | Predicted SF |
|---|---|---|
| **True NY** | 199 | 25 |
| **True SF** | 57 | 211 |

## Key Findings

- `elevation` and `price_per_sqft` are the two strongest single predictors, showing the largest gap in mean/median values between NY and SF listings.
- Grid-search optimization consistently outperforms manually eyeballed thresholds.
- More rules ≠ better accuracy — combining four variables performed worse than the best two/three-variable combination, indicating some rules conflicted or added noise.
- The best model (`elevation`, `price_per_sqft`, `sqft`) reached **87.4% accuracy**, correctly classifying 410 out of 492 listings.

## Possible Next Steps

- Replace manual/grid-search thresholds with a proper decision tree or logistic regression model for comparison.
- Use cross-validation to check whether thresholds generalize beyond this dataset.
- Explore interaction terms (e.g., `price_per_sqft × elevation`) instead of independent thresholds.
