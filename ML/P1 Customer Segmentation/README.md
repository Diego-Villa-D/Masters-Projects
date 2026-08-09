# Customer Segmentation with RFM Analysis (Hierarchical Clustering, K-Means & K-NN)

**Author:** Diego Armando Villa Domínguez
**Course:** Machine Learning Introduction

## Objective

Apply hierarchical clustering, K-Means, and K-NN in an integrated pipeline to segment customers of an online store using **RFM analysis** (Recency, Frequency, Monetary value), and build a classifier that can automatically assign new customers to the appropriate segment.

## Dataset

**[Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)** (UCI Machine Learning Repository) — real transactional data from a UK-based online retailer, covering the period 2010–2011 (541,909 transactions).

### Cleaning pipeline

1. Remove records without a `CustomerID`.
2. Remove cancelled invoices (prefix `"C"`).
3. Filter out non-positive `Quantity`.
4. Filter out non-positive `UnitPrice`.
5. Restrict the analysis to `Country == "United Kingdom"`.

### RFM feature construction

| Variable | Definition |
|---|---|
| **Recency (R)** | Days since the customer's last purchase, relative to a reference date (one day after the last transaction in the dataset). |
| **Frequency (F)** | Number of unique invoices per customer. |
| **Monetary (M)** | Total accumulated spend per customer (`Quantity × UnitPrice`, summed). |

```python
df = pd.read_excel("Online Retail.xlsx")

# --- Cleaning ---
df = df[df["CustomerID"].notna()]
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]
df = df[df["Country"] == "United Kingdom"]

df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# --- Customer-level aggregation ---
fecha_ref = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (fecha_ref - x.max()).days,  # Recency
    "InvoiceNo": "nunique",                                # Frequency
    "TotalPrice": "sum"                                    # Monetary
}).rename(columns={
    "InvoiceDate": "recencia",
    "InvoiceNo": "frecuencia",
    "TotalPrice": "monto"
}).reset_index()
```

## Stage 1 — Preprocessing & Exploratory Analysis

- Histograms, boxplots, a correlation matrix, and a pairplot were built on the raw RFM variables.
- **Correlation:** moderate positive correlation between Frequency and Monetary (0.51); weak negative correlations between Recency and the other two (-0.27 with Frequency, -0.13 with Monetary).
- **Skewness (untransformed):** Recency = 1.25, Frequency = 10.75, Monetary = 20.20 — all heavily right-skewed, driven by a small group of high-value, high-frequency customers.
- A `np.log1p()` transformation was applied to all three variables before scaling with `StandardScaler`, substantially reducing skewness and outlier influence — Monetary in particular approached a near-normal shape after the transform.
- `random_state = 42` was fixed across all stochastic procedures for reproducibility.

## Stage 2 — Hierarchical Clustering (Exploratory)

Dendrograms were built on the scaled data using two linkage criteria (truncated to the last 30 merges for readability):

- **Ward linkage:** suggests a two-cluster structure, based on a pronounced jump in the final merges — though sub-groups are visible within each branch.
- **Complete linkage:** suggests roughly three clusters, forming more internally homogeneous groups (as expected, since Complete linkage minimizes maximum within-cluster distance) — but with one highly degenerate cluster (only 22 members) once cut.

The tree was cut with `fcluster`, and results were visualized on a 2D scatter plot. Hierarchical clustering offered a useful high-level view of the data's structure, but — unlike K-Means — it doesn't scale well to reassigning new customers without recomputing the full linkage.

## Stage 3 — K-Means Segmentation

### Choosing K

- **Elbow method:** inertia drops sharply from K=2 to K=5, then flattens — suggesting K=5 as a reasonable elbow point.
- **Silhouette score:** maximized at K=2, favoring the simplest possible split.
- These two criteria disagree, and neither fully aligns with the hierarchical clustering result from Stage 2. **K=5 was selected explicitly for business reasons** — a 2–3 cluster split was judged too coarse to differentiate meaningfully distinct customer profiles, even though it scores better on the silhouette metric. All five resulting clusters exceeded the 3%-of-total-customers threshold, confirming none were degenerate.

### Segment characterization (original scale)

| Cluster | Recency (days) | Frequency (invoices) | Monetary (£) | % of customers | Label |
|---|---|---|---|---|---|
| 0 | 206.57 | 1.21 | 269.51 | 28.21% | **Dormant customers** |
| 1 | 19.02 | 5.93 | 2,120.19 | 21.91% | **Active customers** |
| 2 | 11.32 | 20.93 | 12,558.09 | 7.42% | **VIP customers** |
| 3 | 26.90 | 1.65 | 381.12 | 19.41% | **New customers** |
| 4 | 102.86 | 3.18 | 1,378.34 | 23.04% | **At-risk customers** |

### On the silhouette metric

Re-running K selection on the **untransformed** RFM variables yields noticeably higher silhouette scores (especially at K=2 and K=3) — but this is misleading. Those high scores come from a handful of extreme outliers in Frequency and Monetary sitting far away from the rest of the data, which inflates between-cluster separation without producing a useful segmentation. The log-transformed version, despite scoring lower on silhouette, produces a segmentation that isn't dominated by a few outlier accounts and is far more actionable for characterizing customer behavior.

### Comparison with hierarchical clustering

A contingency table and the **adjusted Rand index (ARI = 0.106)** were computed between the hierarchical (Complete linkage, 3 clusters) and K-Means (5 clusters) assignments. The low agreement reflects that the two methods are comparing partitions of different granularity (3 vs. 5 clusters) using different linkage criteria — the hierarchical clusters each span multiple K-Means segments, so there's no clean one-to-one correspondence between them.

## Stage 4 — New-Customer Classification with K-NN

The K-Means labels were used as the target variable for a K-NN classifier, trained on an 80/20 stratified split.

- **Hyperparameter search:** values of K = 1, 3, 5, 7, 9... were evaluated via 5-fold stratified cross-validation. **K = 7** achieved the best average accuracy (96.68%).
- **Weighting scheme:** `weights="distance"` outperformed `weights="uniform"` (98.60% vs. 97.83% accuracy) and was used for the final model.
- **Final test performance:** **98.60% accuracy**, with precision, recall, and F1-score above 0.97 for every segment. Misclassifications were rare and concentrated between segments with adjacent profiles (e.g., Active vs. At-risk).

### Demonstration on new customers

Three simulated customers were passed through the **exact same preprocessing chain** used during training (`log1p` → `scaler.transform()`, never `fit_transform()`, to avoid leaking new-customer statistics into the fitted scaler):

| Recency | Frequency | Monetary (£) | Predicted Segment |
|---|---|---|---|
| 220 | 1 | 300 | Dormant customers |
| 15 | 8 | 2,500 | Active customers |
| 8 | 25 | 15,000 | VIP customers |

All three predictions align with the expected business interpretation of each profile.

## Tools

- **Language:** Python
- **Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn (`KMeans`, `AgglomerativeClustering`/`scipy.cluster.hierarchy`, `KNeighborsClassifier`, `StandardScaler`, `PCA`, `silhouette_score`, `adjusted_rand_score`), scipy
