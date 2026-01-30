# EI vs UCB/LCB: Recommendation for Cu₃VS₄ BO

## Executive Summary

**Recommendation: Stick with EI (Expected Improvement)** ✅

Your current EI implementation is well-suited for your goals. However, I'll show you how to add UCB/LCB as an optional alternative if you want more explicit exploration control.

---

## Your Specific Context

1. **Limited data**: 41 experiments, 31 successful
2. **Multi-objective**: Minimize GSD, maximize Squareness
3. **Constrained**: Size targeting + feasibility (product, purity, cubic)
4. **Goal**: Find good synthesis conditions (not pure exploration)
5. **Current implementation**: EI with scalarized utility + constraint probabilities

---

## Why EI is Better for Your Case

### ✅ 1. Constraint Handling

**EI** (your current approach):
```python
score = EI(utility) × p_size × p_feas
```
- Multiplicative constraint handling is standard and principled
- Works naturally with probabilistic constraints
- No need to tune additional parameters

**UCB/LCB**:
```python
# Would need to handle constraints differently
score = UCB(utility) × p_size × p_feas  # Still multiplicative, but...
# OR
score = UCB(utility) + penalty_term  # Less principled
```
- Same multiplicative approach works, but EI is more theoretically grounded for constraints

### ✅ 2. Limited Data (41 experiments)

**EI**:
- Focuses on **exploitation** (finding good points near current best)
- Automatically balances exploration when uncertainty is high
- Better for small datasets where you want to find optima quickly

**UCB/LCB**:
- More **exploration-focused** (explicitly prioritizes uncertainty)
- Requires tuning `beta` parameter carefully
- Can waste experiments exploring when you have limited budget

### ✅ 3. Multi-Objective Scalarization

**Your current approach**:
```python
utility = w_GSD * GSD_norm - w_Squareness * Squareness_norm  # minimize
EI = expected_improvement_min(mu_u, sd_u, best_u)
```

**EI advantages**:
- Works naturally with scalarized utilities
- "Best so far" baseline (`best_u`) is well-defined
- Improvement is a clear metric

**UCB/LCB**:
- Would need: `UCB = mu_u - beta * sd_u` (for minimization)
- Less intuitive: "confidence bound" vs "expected improvement"
- Beta tuning becomes critical

### ✅ 4. You Already Have Exploration

Your code includes:
```python
include_exploration_alt: bool = True
# Adds high-uncertainty feasible candidates
```

This gives you the best of both worlds:
- **EI** for main recommendations (exploitation)
- **High-uncertainty** candidates for exploration

---

## When UCB/LCB Might Be Better

Consider UCB/LCB if:

1. **Very early iterations** (< 10 experiments)
   - Need aggressive exploration
   - EI might be too conservative

2. **Want explicit exploration control**
   - Beta parameter gives you direct control
   - Can adjust exploration vs exploitation balance

3. **Uncertainty is your primary concern**
   - Want to reduce model uncertainty
   - UCB explicitly prioritizes high-uncertainty regions

---

## Performance Comparison (Theoretical)

| Aspect | EI (Current) | UCB/LCB |
|--------|-------------|---------|
| **Constraint handling** | ✅ Excellent (multiplicative) | ⚠️ Good (same approach) |
| **Small datasets** | ✅ Better (exploitation-focused) | ⚠️ Can over-explore |
| **Multi-objective** | ✅ Natural fit | ✅ Works but less intuitive |
| **Parameter tuning** | ✅ None needed | ⚠️ Beta tuning critical |
| **Theoretical foundation** | ✅ Well-established | ✅ Well-established |
| **Your use case** | ✅ **Recommended** | ⚠️ Overkill |

---

## Implementation: Adding UCB/LCB as Alternative

If you want to add UCB/LCB as an optional alternative, here's how:

### Option 1: Add `acquisition_type` Parameter

```python
def score_candidates(
    self,
    Xcand_raw: np.ndarray,
    target_size: float,
    size_tol: float,
    p_size_min: float,
    p_feas_min: float,
    acquisition_type: str = "EI",  # "EI" or "UCB"
    beta_val: float = None,  # Only used for UCB
    iteration: int = 0,  # For adaptive beta
):
    """
    Compute constrained acquisition scores.
    
    acquisition_type: "EI" (default) or "UCB"
    """
    # ... existing prediction code ...
    
    # Compute utility (same for both)
    mu_u = (w_g * mu_g_n) - (w_sq * mu_q_n)
    sd_u = np.sqrt((w_g * sd_g_n) ** 2 + (w_sq * sd_q_n) ** 2)
    
    if acquisition_type == "EI":
        # Current EI approach
        best_u = float(np.min(u_h))  # from historical data
        acq = expected_improvement_min(mu_u, sd_u, best_u)
        
    elif acquisition_type == "UCB":
        # UCB for minimization: mu - beta * sigma
        if beta_val is None:
            beta_val = self.beta(iteration)
        acq = -(mu_u - beta_val * sd_u)  # Negative because we maximize acquisition
        
    else:
        raise ValueError(f"Unknown acquisition_type: {acquisition_type}")
    
    score = acq * p_size * p_feas
    return score, feasible_mask, p_size, p_feas, preds
```

### Option 2: Hybrid Approach (Recommended)

Keep EI as default, but add UCB for early iterations:

```python
def score_candidates(
    self,
    Xcand_raw: np.ndarray,
    target_size: float,
    size_tol: float,
    p_size_min: float,
    p_feas_min: float,
    use_ucb_early: bool = True,  # Use UCB for first 5 iterations
    iteration: int = 0,
):
    # ... existing code ...
    
    # Choose acquisition based on iteration
    if use_ucb_early and iteration < 5:
        # Early exploration with UCB
        beta_val = self.beta(iteration)
        acq = -(mu_u - beta_val * sd_u)
    else:
        # Standard EI (exploitation)
        best_u = float(np.min(u_h))
        acq = expected_improvement_min(mu_u, sd_u, best_u)
    
    score = acq * p_size * p_feas
    return score, feasible_mask, p_size, p_feas, preds
```

---

## Empirical Recommendation

**For your specific case:**

1. **Stick with EI** as your primary acquisition function ✅
2. **Keep `include_exploration_alt=True`** for exploration ✅
3. **Consider UCB only if**:
   - You're starting with < 10 experiments
   - You want to explicitly explore certain regions
   - You're doing a systematic comparison study

**Why?**
- Your dataset is small (41 experiments)
- You want to find good conditions (exploitation)
- EI is theoretically sound and works well
- You already have exploration via `include_exploration_alt`

---

## Beta Parameter Tuning (If You Use UCB)

If you do implement UCB, your current `beta()` method:

```python
def beta(self, t: int, beta_start: float = 2.0, beta_growth: float = 0.05) -> float:
    return beta_start * (1.0 + beta_growth * t)
```

**Recommendations:**
- **Early iterations** (t < 5): `beta_start = 2.0-3.0` (more exploration)
- **Later iterations** (t > 10): `beta_start = 1.0-1.5` (more exploitation)
- **Your current schedule**: `beta_growth = 0.05` is reasonable

**Better adaptive schedule:**
```python
def beta(self, t: int, beta_start: float = 2.0) -> float:
    """Decay beta over time (more exploration early, less later)."""
    return beta_start * np.exp(-0.1 * t)  # Exponential decay
```

---

## Code Changes Needed (If Adding UCB)

**Minimal changes:**
1. Add `acquisition_type` parameter to `score_candidates()`
2. Add conditional logic for EI vs UCB
3. Update `recommend_for_size()` to pass `acquisition_type`

**No changes needed:**
- GP models (same predictions)
- Constraint handling (same approach)
- Utility computation (same)

---

## Final Recommendation

**✅ Use EI (your current implementation)**

**Reasons:**
1. Well-suited for constrained optimization
2. Better for small datasets (exploitation-focused)
3. Works naturally with your multi-objective scalarization
4. No parameter tuning needed
5. You already have exploration via `include_exploration_alt`

**Consider UCB only if:**
- You're doing a systematic comparison
- You want explicit exploration control
- You're starting with very few experiments (< 10)

**Bottom line:** Your current EI implementation is theoretically sound and appropriate for your goals. The `beta()` method you have is fine to keep for potential future use, but you don't need to switch to UCB/LCB.

---

## References

- **EI for constrained BO**: Gelbart et al. (2014) - "Bayesian Optimization with Unknown Constraints"
- **Multi-objective EI**: Knowles (2006) - "ParEGO: A hybrid algorithm with on-line landscape approximation"
- **UCB vs EI**: Frazier (2018) - "A Tutorial on Bayesian Optimization"
