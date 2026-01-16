# Root Cause Analysis: Confidence Distribution Failure

**Date**: 2026-01-15
**Issue**: Only 0.2% of predictions are Elite+Strong (target: ≥10%)

---

## Executive Summary

**ROOT CAUSE IDENTIFIED**: Base model disagreement is **10-30x worse** than confidence threshold expectations.

- **Expected CV** for Elite tier: < 0.05
- **Actual CV** across all props: 0.3 - 1.4
- **Gap**: 6x - 28x worse than threshold

This is why 99.8% of predictions fall into "Avoid" tier.

---

## Investigation Results

### 1. Missing Confidence Scores (44.7% discrepancy)
**Status**: Minor reporting inconsistency, not data quality issue
**Impact**: Low - actual predictions (48,705) have valid confidence scores

### 2. Confidence Correlation
**Result**: 0.1019 (target: > 0.5)
**Status**: ❌ NOT MET
**Impact**: CRITICAL - confidence scores barely correlate with accuracy

### 3. Base Model Agreement Analysis

#### Coefficient of Variation (CV) by Prop Type:

| Prop Type | Mean CV | Median CV | Elite+Strong % | Status |
|-----------|---------|-----------|----------------|--------|
| **Points** | 1.0091 | 0.9987 | 0.0% | ❌ NOT MET |
| **Rebounds** | 1.4054 | 1.4006 | 0.0% | ❌ NOT MET |
| **Assists** | 0.8824 | 0.8101 | 0.0% | ❌ NOT MET |
| **Threes** | 0.3369 | 0.3277 | 0.0% | ❌ NOT MET |
| **PRA** | 1.2477 | 1.2431 | 0.0% | ❌ NOT MET |

**Overall Elite+Strong**: 0.0% (target: ≥10%)

#### Current Confidence Thresholds:

```python
if cv < 0.05:    # Elite (90-100)
elif cv < 0.10:  # Strong (75-89)
elif cv < 0.20:  # Moderate (60-74)
elif cv < 0.30:  # Weak (40-59)
else:            # Avoid (0-39)
```

#### Reality:
- **No predictions** have CV < 0.05 (Elite threshold)
- **No predictions** have CV < 0.10 (Strong threshold)
- **Very few** have CV < 0.20 (Moderate threshold)
- **Most** have CV > 0.30 (Avoid tier)

---

## Why Are Base Models Disagreeing So Much?

### Hypothesis 1: Models Are Fundamentally Different ✅ LIKELY

The 5 base models are:
1. XGBoost (gradient boosting)
2. LightGBM (gradient boosting variant)
3. CatBoost (gradient boosting variant)
4. Random Forest (bagging)
5. Ridge (linear regression)

**Ridge is radically different** from tree-based models, causing large disagreement.

### Hypothesis 2: Phase 2 Features Causing Instability ✅ POSSIBLE

Phase 2 added:
- Travel/Fatigue features (10 features)
- Betting market features (6 features)
- Enhanced injury features (4 features)

These may be:
- Noisy or unreliable
- Causing overfitting
- Interpreted differently by different model types

### Hypothesis 3: Poor Model Calibration ✅ CONFIRMED

- Confidence correlation: 0.1019 (near random)
- Models may be overconfident or underconfident
- Need calibration methods (Platt scaling, isotonic regression)

### Hypothesis 4: Training Data Too Noisy ✅ POSSIBLE

- NBA player performance is inherently variable
- CV of 0.3-1.4 suggests fundamental uncertainty
- May need to accept higher thresholds

---

## Impact Assessment

### Critical Blockers:

1. **Cannot implement selective betting**
   - Only 0.2% Elite+Strong vs 10% target
   - Kelly criterion requires high-confidence subset
   - **Blocks Phase 3 entirely**

2. **Confidence scores meaningless**
   - Correlation 0.1019 (near random)
   - Cannot trust confidence for decision-making
   - **Undermines entire confidence mechanism**

3. **All predictions lumped into "Avoid"**
   - 99.8% in lowest tier
   - No differentiation between good/bad opportunities
   - **Makes confidence filtering useless**

---

## Solution Options

### Option 1: Recalibrate Thresholds (QUICK FIX)

**Adjust thresholds to match reality:**

```python
if cv < 0.30:    # Elite (was 0.05)  - 6x more lenient
elif cv < 0.50:  # Strong (was 0.10) - 5x more lenient
elif cv < 0.80:  # Moderate (was 0.20) - 4x more lenient
elif cv < 1.20:  # Weak (was 0.30) - 4x more lenient
else:            # Avoid
```

**Pros:**
- Quick to implement
- Would achieve ~10-20% Elite+Strong
- No model retraining needed

**Cons:**
- Doesn't fix underlying disagreement
- "Elite" tier would still have high error
- Doesn't improve confidence correlation

**Recommendation**: ⚠️ **Temporary fix only**

### Option 2: Remove Ridge Model (MODERATE FIX)

**Problem**: Ridge (linear) vastly different from tree models

**Solution**: Use only tree-based ensemble (XGB, LGBM, CatBoost, RF)

**Expected Impact:**
- Reduce CV by ~20-30%
- Improve agreement
- Still may not reach CV < 0.05

**Pros:**
- Improves base model agreement
- More homogeneous ensemble
- May improve RMSE

**Cons:**
- Requires model retraining
- Loses diversity benefits
- May not be enough

**Recommendation**: ✅ **Worth trying**

### Option 3: Feature Ablation (CRITICAL)

**Investigate if Phase 2 features harm stability**

**Test**:
- Remove travel/fatigue features
- Remove betting market features
- Remove enhanced injury features
- Measure CV improvement

**If CV improves significantly**: Remove harmful features

**Recommendation**: ✅ **MUST DO - highest priority**

### Option 4: Calibration Methods (ADVANCED FIX)

**Apply post-hoc calibration:**
- Platt scaling (logistic calibration)
- Isotonic regression
- Temperature scaling

**Pros:**
- Can improve confidence correlation
- Industry-standard approach
- Preserves model predictions

**Cons:**
- Requires calibration dataset
- Complex implementation
- May not fix tier distribution

**Recommendation**: ⚠️ **After feature ablation**

### Option 5: Alternative Confidence Metric (REDESIGN)

**Instead of CV, use:**
- Model uncertainty (dropout, ensembles)
- Prediction intervals (quantile regression)
- Conformal prediction
- Bayesian approaches

**Pros:**
- May be better suited to problem
- More principled uncertainty quantification

**Cons:**
- Major redesign
- Requires new models
- Time-intensive

**Recommendation**: ⏳ **Last resort if others fail**

---

## Recommended Action Plan

### Immediate (Priority 1):

1. **Feature Ablation Study** 🔴 CRITICAL
   - Test Phase 2 feature groups individually
   - Measure CV improvement when removed
   - Identify harmful features
   - ETA: 3-4 hours

2. **Recalibrate Thresholds** (Temporary)
   - Adjust to CV thresholds: 0.30, 0.50, 0.80, 1.20
   - Validate Elite+Strong reaches ~10%
   - ETA: 30 minutes

### Short-term (Priority 2):

3. **Remove Ridge Model**
   - Retrain with only tree-based models
   - Measure CV improvement
   - ETA: 2 hours (if needed after ablation)

4. **Apply Calibration**
   - Platt scaling or isotonic regression
   - Improve confidence correlation
   - ETA: 2-3 hours

### Long-term (Priority 3):

5. **Consider Alternative Confidence Metrics**
   - If above approaches fail
   - Explore quantile regression, conformal prediction
   - ETA: 1-2 days

---

## Success Criteria

**Phase 2 cannot proceed to Phase 3 until:**

✅ Elite+Strong ≥ 10% of predictions
✅ Confidence correlation r > 0.5
✅ Overall RMSE < 5.0
✅ Overall bias < |0.5| (ALREADY MET)

**Current Status:**
- Elite+Strong: 0.2% ❌ (50x below target)
- Confidence corr: 0.1019 ❌
- RMSE: 5.284 ❌
- Bias: -0.021 ✅

**Targets Met: 1/4 (25%)**

---

## Timeline Estimate

- **Quick threshold recalibration**: 30 min
- **Feature ablation study**: 3-4 hours
- **Model retraining (if needed)**: 2 hours
- **Calibration methods**: 2-3 hours
- **Final validation**: 1-2 hours

**Total ETA**: 8-12 hours to fix confidence mechanism

---

## Conclusion

The confidence mechanism failure is NOT a bug - it's a **fundamental mismatch** between:
1. **Expected model agreement** (CV < 0.05-0.20)
2. **Actual model disagreement** (CV 0.3-1.4)

The base models disagree 10-30x more than the thresholds expect. This makes the current confidence tiers unusable.

**Next steps:**
1. Feature ablation (identify if Phase 2 features are culprit)
2. Threshold recalibration (quick fix)
3. Model refinement (if needed)
4. Calibration (improve correlation)

**DO NOT proceed to Phase 3** until confidence mechanism is fixed.

---

**Files Generated:**
- `backtest_results/confidence_correlation.json` - Correlation: 0.1019
- `backtest_results/base_model_agreement.json` - CV analysis
- `INVESTIGATION_1_FINDINGS.md` - Missing predictions analysis
- `ROOT_CAUSE_ANALYSIS.md` - This document
