# Why R² is Negative: Understanding the ARMA(1,1) Overfitting Problem

## The Problem

When fitting ARMA(1,1) to Boston electricity load data, we get **R² = -2.61**, which is negative and seems impossible. Here's what's actually happening:

## Root Cause Analysis

### The Numbers Don't Lie

```
Original Data (y):
  Variance: 153,739 MWh²
  Std Dev: 392 MWh
  Range: 1,583 - 4,340 MWh

Fitted Values (y_pred from SARIMAX):
  Variance: 538,643 MWh² (!!)  <-- 3.5x larger than original!
  Std Dev: 734 MWh
  Range: 547 - 3,675 MWh

Residuals (y - y_pred):
  Variance: 554,367 MWh² (!!)  <-- 3.6x larger than original!
  Std Dev: 745 MWh
  Range: -1,408 to +2,975 MWh

R² = 1 - (554,367 / 153,739) = 1 - 3.61 = -2.61
```

### What This Means

**The fitted values have MORE variance than the original data!** This is the hallmark of severe overfitting. The model is introducing noise rather than reducing it.

## Why This Happens

### 1. **Near-Cancellation of AR and MA Terms**

The SARIMAX converged with:
- **AR(1) coefficient**: φ₁ = 0.869
- **MA(1) coefficient**: θ₁ = -0.901

These coefficients are nearly equal and opposite! This creates a **near-cancellation** problem:

```
ARMA(1,1): y[t] = φ₁·y[t-1] + ε[t] + θ₁·ε[t-1]
         = 0.869·y[t-1] + ε[t] - 0.901·ε[t-1]
```

When φ₁ ≈ -θ₁, the model becomes numerically unstable. Small estimation errors get amplified, creating wild predictions.

### 2. **Parameter Redundancy**

With seasonal harmonics + day-of-week dummies already in the model:
- The **AR term** tries to capture persistence (yesterday → today)
- The **MA term** tries to capture shock propagation
- But these effects are ALREADY captured by the exogenous variables!

This creates **parameter redundancy**: multiple parameters trying to explain the same variance, leading to:
- Unstable parameter estimates
- Overfitting
- Poor out-of-sample predictions

### 3. **The Variance Explosion**

In ARMA models, variance of predictions is:
```
Var(y_pred) = function of (AR coefs, MA coefs, innovation variance)
```

When AR and MA terms nearly cancel:
- The model becomes extremely sensitive to parameter values
- Small changes in data → large changes in predictions
- Fitted values oscillate wildly
- Residuals become larger than the signal itself

## Mathematical Explanation

### Standard R² Formula

```
R² = 1 - SS_res / SS_tot
   = 1 - Σ(y - ŷ)² / Σ(y - ȳ)²
   = 1 - var(residuals) / var(y)
```

### Why R² Can Be Negative

R² is **NOT** the square of correlation! It can be negative when:

```
SS_res > SS_tot
⟺ Σ(y - ŷ)² > Σ(y - ȳ)²
⟺ var(residuals) > var(y)
```

This means: **Your model predictions are worse than just predicting the mean!**

### In Our Case

```
var(residuals) = 554,367
var(y) = 153,739

554,367 / 153,739 = 3.61

R² = 1 - 3.61 = -2.61
```

The model is **3.6 times worse** than the naive mean prediction.

## Why AR(1) Works But ARMA(1,1) Fails

### AR(1) Model (Works Well)
```
y[t] = c + φ₁·y[t-1] + ε[t]
```
- **Simple**: One parameter to estimate
- **Stable**: No parameter interactions
- **R² = 0.76**: Good fit

### ARMA(1,1) Model (Fails)
```
y[t] = c + φ₁·y[t-1] + ε[t] + θ₁·ε[t-1]
```
- **Complex**: Two interacting parameters
- **Unstable**: Near-cancellation (φ₁ ≈ -θ₁)
- **R² = -2.61**: Catastrophic failure

## The "Convergence" Paradox

You might ask: "But the model converged! How can it be so bad?"

**Answer**: Convergence means the optimization algorithm found a local minimum, NOT that the model is good:

1. SARIMAX maximized the likelihood
2. But the likelihood function has multiple local maxima
3. The algorithm found a "solution" that's numerically valid but statistically meaningless
4. The MA term is trying to "fix" problems caused by overfitting the AR term

## Implications for Model Selection

### AIC vs R² Paradox

Notice this interesting pattern:

| Model | R² | AIC | BIC |
|-------|-----|-----|-----|
| AR(1) | 0.76 | 43,165 | 43,281 |
| ARMA(1,1) | -2.61 | 54,187 | 54,309 |

**Higher AIC means worse fit**, even though both "converged"!

### Why AIC Catches This

AIC = 2k - 2·ln(L)

Where:
- k = number of parameters
- L = likelihood

Even though ARMA(1,1) maximized likelihood locally, the overall likelihood is much worse because:
- The model explains the data poorly (low L)
- It uses more parameters (higher k penalty)

## How to Prevent This

### 1. **Always Check R²**
```python
if fit['r_squared'] < 0:
    print("⚠️  Model is overfitting! Residuals worse than naive mean.")
```

### 2. **Compare Multiple Metrics**
```python
# Don't rely on convergence alone
print(f"Converged: {fit['arma_params']['converged']}")
print(f"R²: {fit['r_squared']:.4f}")  # Should be positive!
print(f"AIC: {fit['aic']:.2f}")        # Lower is better
```

### 3. **Check Parameter Values**
```python
ar_coef = fit['arma_params']['ar_coefs'][0]
ma_coef = fit['arma_params']['ma_coefs'][0]

if abs(ar_coef + ma_coef) < 0.1:  # Near cancellation
    print("⚠️  AR and MA coefficients nearly cancel!")
```

### 4. **Visual Inspection**
```python
plt.scatter(fit['y'], fit['y_pred'])
plt.plot([fit['y'].min(), fit['y'].max()], 
         [fit['y'].min(), fit['y'].max()], 'r--')
plt.xlabel('Actual')
plt.ylabel('Predicted')
# Points should cluster around the diagonal
```

## Lessons Learned

### 1. **Simpler is Often Better**
- AR(1): 1 parameter, R² = 0.76 ✓
- ARMA(1,1): 2 parameters, R² = -2.61 ✗

### 2. **Check ALL Diagnostics**
Don't just trust:
- ❌ Convergence status alone
- ✓ R², AIC, BIC, residual plots, parameter values

### 3. **Domain Knowledge Matters**
Electricity load is naturally AR(1):
- Today depends on yesterday
- No evidence of MA effects (shocks don't propagate)
- Day-of-week already captures weekly patterns

### 4. **Parameter Parsimony**
With rich exogenous variables (harmonics + day-of-week), you need LESS autocorrelation modeling, not more.

## Practical Recommendation

**For electricity load data:**

```python
# ✓ GOOD: Simple and stable
fit = fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=0)

# ✗ BAD: Overfits and unstable
fit = fit_seasonal_garch_model(srs, n_harmonics=3, ar_order=1, ma_order=1)
```

## References

- Box-Jenkins methodology: "Identification, estimation, and diagnostic checking"
- Burnham & Anderson (2002): Model Selection and Multimodel Inference
- "Why R² can be negative": https://stats.stackexchange.com/q/12900

---

## Summary

**Negative R² is NOT a bug—it's a feature!** It's the model telling you loud and clear:

> "I'm fitting so poorly that you'd be better off just predicting the mean every time!"

In this case, ARMA(1,1) introduces parameter redundancy and near-cancellation that causes fitted values to be more variable than the original data. The solution is to use the simpler AR(1) model.
