# Generalization, Overfitting & Model Complexity (Polynomial Regression)

**Author:** Diego Armando Villa Domínguez
**Course:** MIACD — Generalization Assignment

## Objective

The goal of this practice is to empirically observe the effects of **overfitting** and the **estimation of generalization error**, by building polynomial regression models of varying complexity.

## Part 1 — Statements on Generalization

Before the practical exercise, several theoretical statements about generalization were discussed and evaluated:

**1. "In real-world problems, the generalization error cannot be estimated exactly."**
Agreed. Generalization error reflects a model's performance on completely new data drawn from the true underlying distribution — data that is, by definition, unknown during training. In practice we only ever have a finite sample, which may not perfectly represent the full population. Because of this, the true generalization error can never be computed exactly; it can only be **estimated** using statistical techniques such as train/test splits, cross-validation, or bootstrap.

**2. "Training error is always optimistic."**
Agreed. Training error is computed on the exact same data the model was fit on — the algorithm has already "seen" these examples and optimized its parameters to minimize error on them. As a result, training error tends to underestimate the error the model will get on new data, making it an optimistic estimate of real-world performance.

**3. "Validation strategies give us an idea of what the generalization error will be."**
Agreed. Validation strategies attempt to simulate model behavior on data that was not involved in training. Methods like k-fold cross-validation, hold-out validation, and bootstrap don't give the exact generalization error, but they do provide a statistically grounded estimate of it.

**4. "Trying to improve training performance by increasing model complexity can lead to overfitting — training error becomes misleading."**
Strongly agreed. Overfitting is one of the central risks in machine learning: as model complexity increases, the model may start learning noise and dataset-specific quirks rather than the true underlying pattern. Training error generally keeps decreasing as complexity increases, but this doesn't mean the model generalizes better — in fact, the opposite often happens: training error keeps falling while validation error starts to rise.

## Part 2 — Practical Exercise

### Synthetic Dataset

A synthetic dataset was generated from the following ground-truth model:

```
y = 2 + 4(5x - 1)(4x - 2)(x - 0.8)² + ε
```

where `x` ranges between 0.1 and 1, and `ε` is Gaussian noise with mean 0 and standard deviation 0.5.

Two functions were implemented in Python: a **generator** function that produces noisy observed values, and a **true function** with the noise removed, used to visualize the underlying shape of the relationship.

```python
# Generator function
np.random.seed(42)

def generador(n=200, sigma=0.5):
    x = np.random.uniform(0.1, 1, n)
    epsilon = np.random.normal(0, sigma, n)
    y = 2 + (4*(5*x-1))*(4*x-2)*(x-0.8)**2 + epsilon
    return x, y

X, y = generador(n=200, sigma=0.5)

# True function (no noise)
def funcion_verdadera(x):
    return 2 + (4*(5*x-1))*(4*x-2)*(x-0.8)**2
```

The dataset was split into:
- **60%** training
- **20%** validation
- **20%** test

### Polynomial Models of Increasing Complexity

Nine polynomial regression models were trained, using odd degrees from **1 to 17**. Validation MSE for each:

| Degree | Validation MSE |
|---|---|
| 1 | 0.457 |
| 3 | 0.352 |
| 5 | 0.278 |
| 7 | 0.292 |
| 9 | 0.299 |
| 11 | 0.325 |
| 13 | 0.323 |
| 15 | 0.321 |
| 17 | 0.339 |

**Observations:**
- **Degrees 1 and 3** show clear **underfitting** — the model is too simple to capture the true curve.
- **Degrees 5 and 7** achieve the best fit, with degree 5 producing the lowest validation error.
- **From degree 9 onward**, models start to **overfit**: training error keeps dropping, but validation and test error plateau or increase.

This pattern is confirmed in the "Error vs. Model Complexity" plot: training error decreases monotonically with degree, while validation and test error bottom out around degree 5 and creep back up afterward — the textbook signature of the bias-variance tradeoff.

### Cross-Validation

K-fold cross-validation (k=5) was implemented to estimate the generalization error of each polynomial model, and compared against the simple hold-out validation results.

**Findings:**
- Both methods show the same overall pattern: error decreases from degree 1 to degree 5 (underfitting region), then increases or plateaus from degree 5 onward (overfitting region).
- Cross-validation produced slightly **lower and more stable** average errors than hold-out validation across all degrees, since it uses the available data more efficiently — training and evaluating across multiple partitions of the dataset instead of a single split.

### Effect of Sample Size (Learning Curves)

Learning curves were built for a simple model (degree 5) and a complex model (degree 11), training each on progressively larger subsets of data and plotting training/validation error against training set size.

**Q: How does the gap between training and validation error evolve as the number of observations increases?**
A: In both models, the gap tends to narrow slightly as the training set grows. Training error increases somewhat, while validation error stays roughly stable or decreases slightly — indicating improved generalization.

**Q: Which model benefits more from additional data?**
A: The **degree-11 model** benefits more. Its training error rises and the gap with validation error shrinks as more data becomes available, showing that overfitting is reduced when the training set grows.

**Q: What are the implications for real-world problems where data is limited?**
A: When data is scarce, highly complex models are more prone to overfitting and tend to generalize worse. In these situations it's generally preferable to use a moderately complex model (like degree 5) or to collect more data so that more complex models can use their extra capacity without sacrificing generalization.

## Theoretical–Practical Integration

The experimental results support the theoretical statements discussed earlier:

- **Training error is optimistic** — it decreased continuously as polynomial degree increased, even well past the point where the model stopped generalizing well.
- **Validation error was minimized at degree 5** and increased for higher degrees, confirming that increasing model complexity can induce overfitting: degrees 9 and 11 achieved lower training error but higher validation/test error.
- **Both hold-out and cross-validation identified degree 5** as the model with the best generalization capacity, reinforcing that validation strategies — despite not giving an exact generalization error — provide a reliable estimate of it.
