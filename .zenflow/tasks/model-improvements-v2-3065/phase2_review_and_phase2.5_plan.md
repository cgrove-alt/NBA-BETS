# Phase 2 Review & Phase 2.5 Debugging Sprint Plan

**Date**: January 15, 2026
**Reviewer**: User
**Status**: CRITICAL ISSUES IDENTIFIED - PHASE 2.5 REQUIRED

---

## Executive Summary

Task 2.6 (Phase 2 Backtest with Confidence Filtering) was **successfully implemented** but revealed **critical foundational issues** that block progression to Phase 3. The confidence filtering mechanism works as designed, but 90.36% of predictions fall into the "avoid" tier (confidence < 40), making the system unusable for betting.

**Decision**: Implement **Phase 2.5: Model Calibration Sprint** before proceeding to Phase 3.

---

## 🚨 Critical Findings

### 1. **Extreme Confidence Distribution Problem** (BLOCKER)

**Current State**:
- **Elite tier** (90-100): 39 predictions (0.08%)
- **Strong tier** (75-89): 133 predictions (0.27%)
- **Elite + Strong**: 172 predictions (**0.25%** of total)
- **Avoid tier** (<40): 44,011 predictions (**90.36%**)

**Expected State**:
- Elite + Strong should be **10-20%** of predictions
- Avoid tier should be **<50%**

**Root Cause**:
- Base models (5 models: XGBoost, LightGBM, CatBoost, RandomForest, Ridge) have **very high disagreement**
- Coefficient of variation (CV) > 0.20 for 90% of predictions
- 86% of sampled predictions hit the **minimum confidence floor (30)**

**Impact**: Only 0.25% of predictions are actionable for betting (too restrictive for viable strategy)

---

### 2. **Model Performance Regression** (BLOCKER)

| Metric | Phase 1 | Phase 2 | Change | Status |
|--------|---------|---------|--------|--------|
| **Overall RMSE** | 5.435 | 5.707 | +4.8% | ❌ WORSE |
| **Overall MAE** | 3.557 | 3.735 | +5.0% | ❌ WORSE |
| **Overall R²** | 0.681 | 0.643 | -5.6% | ❌ WORSE |
| **Overall Bias** | -0.601 | -1.671 | +178% | ❌ WORSE |

**Analysis**:
- Phase 2 added 224 more games (596 vs 372), but this doesn't explain 4.8% decline
- Phase 2 features may be **introducing noise** rather than signal
- Systematic bias nearly **tripled** (-0.601 → -1.671)

**Impact**: Phase 2 enhancements made the model worse, not better

---

### 3. **Systematic Underprediction Bias** (HIGH PRIORITY)

**Finding**: Bias of **-1.671** means model consistently predicts 1.7 units too low

**Examples**:
- Points: Predicting ~1.7 points too low on average
- PRA: Predicting ~1.7 combined stats too low

**Impact**:
- Exploitable by opponents (always bet over)
- Suggests miscalibration or missing features
- Bias increased 178% from Phase 1

**Immediate Fix**: Apply bias correction (+1.7) and re-run backtest

---

### 4. **Incomplete Target Validation** (MISSING WORK)

| Target | Goal | Status | Notes |
|--------|------|--------|-------|
| Overall RMSE < 5.0 | Phase 2 | ❌ **NOT MET** | 5.707 (all), 2.122 (filtered) |
| ROI (Elite) > 5% | Phase 2 | ⏳ **NOT TESTED** | Requires odds data |
| Positive CLV | Phase 2 | ⏳ **NOT TESTED** | Requires odds data |
| **Confidence correlation (r > 0.5)** | Phase 2 | ❌ **NOT CALCULATED** | **Should have been done** |

**Missing Analysis**:
1. **Confidence correlation**: Data was available but not calculated
   ```python
   from scipy.stats import pearsonr
   errors = [abs(p.predicted - p.actual) for p in predictions]
   confidences = [p.confidence for p in predictions]
   correlation, p_value = pearsonr(confidences, errors)
   # ^ This should have been computed
   ```

2. **Calibration curves**: Expected vs actual accuracy by confidence bin

3. **Feature ablation**: Which Phase 2 features help vs hurt?

---

## ✅ What Worked

### 1. **Confidence Filtering Mechanism Works**
When the model IS confident, it performs **exceptionally well**:

| Tier | RMSE | Status |
|------|------|--------|
| **Strong** (75-89) | **1.905** | ✅ Excellent |
| **Elite** (90-100) | 2.736 | ✅ Good |
| **Moderate** (60-74) | 2.207 | ✅ Good |

**Key Insight**: Filtered RMSE of 2.122 is **62.8% better** than overall (5.707)

### 2. **Implementation Quality**
- ✅ Clean code architecture (extends SeasonBacktester)
- ✅ Type-safe with dataclasses
- ✅ Comprehensive documentation
- ✅ Follows existing patterns

### 3. **Honest Analysis**
- Agent correctly identified critical issues
- Didn't hide bad results
- Proposed actionable recommendations

---

## 🔍 Root Cause Analysis

### Why Are Base Models Disagreeing So Much?

**Hypothesis 1: Phase 2 Features Introduce Noise**
- Travel/fatigue features (Task 2.1): 10 features
- Betting market features (Task 2.2): 6 features
- Enhanced injury features (Task 2.3): 4 features
- **Total**: 20 new features

**Test**: Ablation study to isolate impact

**Hypothesis 2: Base Models Trained on Different Distributions**
- Models may have different preprocessing
- Different handling of missing values
- Different feature subsets

**Test**: Verify training pipeline consistency

**Hypothesis 3: Model Miscalibration**
- Models overfit to training data
- Poor regularization
- Need retraining with better hyperparameters

**Test**: Cross-validation during training

**Hypothesis 4: Data Quality Issues**
- Missing values in Phase 2 features
- Temporal leakage
- Recent trends (2025-26 season) differ from training data

**Test**: Data quality audit

---

## 📋 Phase 2.5: Model Calibration Sprint

**Duration**: 1 week (5-7 days)
**Goal**: Fix foundational issues before Phase 3

---

### **Task 2.5.1: Calculate Missing Metrics** ⏳
**Priority**: P0 (Critical - Complete Phase 2 validation)
**Estimated Effort**: 2 hours

**Implementation Steps**:
1. Calculate confidence correlation (Pearson r)
   ```python
   errors = [abs(p.predicted - p.actual) for p in predictions]
   confidences = [p.confidence for p in predictions]
   correlation, p_value = pearsonr(confidences, [-e for e in errors])
   # Expect: r > 0.5 (high confidence → low error)
   ```

2. Generate calibration curve
   - Bin predictions by confidence (10-point bins)
   - Calculate actual accuracy per bin
   - Plot: Expected confidence vs Actual accuracy
   - Perfect calibration: 45° line

3. Confidence vs Error scatter plot
   - X-axis: Confidence score
   - Y-axis: Absolute error
   - Add trend line
   - Identify outliers (high confidence but high error)

**Output**:
- `backtest_results/phase2_confidence_analysis.json`
- `backtest_results/calibration_curve.png`

---

### **Task 2.5.2: Apply Bias Correction** ⏳
**Priority**: P0 (Critical - Quick win)
**Estimated Effort**: 1 hour

**Implementation Steps**:
1. Update `comprehensive_backtest.py` BIAS_CORRECTIONS:
   ```python
   BIAS_CORRECTIONS = {
       'points': 1.671,      # Current overall bias
       'rebounds': 0.0,      # TBD based on prop-specific analysis
       'assists': 0.0,       # TBD
       'threes': 0.0,        # TBD
       'pra': 1.671,         # Use overall bias
   }
   ```

2. Calculate prop-specific bias from Phase 2 results

3. Re-run backtest with bias correction

4. Compare results:
   - Phase 2 (no correction): RMSE 5.707, Bias -1.671
   - Phase 2 (corrected): RMSE ?, Bias ~0

**Success Criteria**:
- Overall bias < |0.5|
- RMSE improvement expected (bias contributes to RMSE)

---

### **Task 2.5.3: Feature Ablation Study** ⏳
**Priority**: P0 (Critical - Identify problem features)
**Estimated Effort**: 6 hours

**Implementation Steps**:
1. Create `feature_ablation_backtest.py`

2. Run 5 backtest scenarios (100 games each for speed):
   - **Baseline**: Phase 1 features only (no Phase 2 additions)
   - **+Travel**: Baseline + travel_fatigue.py features
   - **+Betting**: Baseline + betting_market_features.py
   - **+Injury**: Baseline + injury_tracker_v3.py enhancements
   - **All Phase 2**: All features combined

3. For each scenario, measure:
   - Overall RMSE
   - Overall Bias
   - Confidence distribution (% Elite+Strong)
   - Base model agreement (avg CV)

4. Compare results:
   ```
   | Scenario | RMSE | Bias | Elite+Strong % | Avg CV |
   |----------|------|------|----------------|--------|
   | Baseline | ?    | ?    | ?              | ?      |
   | +Travel  | ?    | ?    | ?              | ?      |
   | +Betting | ?    | ?    | ?              | ?      |
   | +Injury  | ?    | ?    | ?              | ?      |
   | All P2   | ?    | ?    | ?              | ?      |
   ```

**Success Criteria**:
- Identify which features improve vs hurt performance
- Remove features that increase CV (model disagreement)
- Keep features that improve RMSE without sacrificing confidence distribution

---

### **Task 2.5.4: Model Recalibration Analysis** ⏳
**Priority**: P1 (High - Understand base model issues)
**Estimated Effort**: 4 hours

**Implementation Steps**:
1. Analyze base model predictions:
   - Load Phase 2 backtest intermediate results
   - For each prediction, extract all 5 base model predictions
   - Calculate per-model RMSE, bias, correlation

2. Identify problematic models:
   - Which model(s) have highest disagreement?
   - Which model(s) have highest bias?
   - Are some models consistently worse?

3. Check ensemble weights:
   - Review `model_weights` in saved model files
   - Are weights balanced or dominated by one model?
   - Meta-learner should learn to downweight bad models

4. Verify training consistency:
   - Check training logs (if available)
   - Verify all models trained on same data
   - Check for different preprocessing

**Output**:
- `backtest_results/base_model_analysis.json`
- Recommendations for model retraining

---

### **Task 2.5.5: Adjust Confidence Thresholds** ⏳
**Priority**: P2 (Medium - Temporary fix until models improve)
**Estimated Effort**: 2 hours

**Implementation Steps**:
1. Analyze current CV distribution:
   ```python
   # Current thresholds:
   Elite:    CV < 0.05  (0.08% of predictions)
   Strong:   CV < 0.10  (0.27%)
   Moderate: CV < 0.20  (3.06%)
   Weak:     CV < 0.30  (6.23%)
   Avoid:    CV >= 0.30 (90.36%)
   ```

2. Calculate percentile-based thresholds:
   ```python
   # Target: 10-20% Elite+Strong
   cv_10th_percentile = ?
   cv_20th_percentile = ?

   # Option A: Loosen thresholds
   Elite:    CV < cv_10th
   Strong:   CV < cv_20th

   # Option B: Use absolute performance (RMSE)
   Elite:    RMSE < 2.5
   Strong:   RMSE < 3.5
   ```

3. Re-run backtest with adjusted thresholds

4. Compare confidence distribution and accuracy

**Caution**: This is a **band-aid** solution. Root cause is model disagreement, not threshold calibration.

**Success Criteria**:
- Elite + Strong: 10-20% of predictions
- Filtered RMSE still < 3.0

---

### **Task 2.5.6: Model Retraining (If Needed)** ⏳
**Priority**: P1 (High - Only if ablation/recalibration don't fix)
**Estimated Effort**: 8-12 hours

**Implementation Steps**:
1. **Only proceed if**:
   - Ablation study shows models are problem (not features)
   - Recalibration analysis identifies systematic issues
   - Bias correction and threshold adjustment insufficient

2. Retrain with improvements:
   - Apply bias correction **during training**
   - Use cross-validation to ensure consistency
   - Better regularization (reduce overfitting)
   - Larger/more recent training dataset
   - Verify base models have reasonable agreement

3. Validate new models:
   - Cross-validation RMSE < 5.0
   - Base model CV < 0.15 on validation set
   - No systematic bias (|bias| < 0.5)

**Success Criteria**:
- Overall RMSE < 5.0
- Elite + Strong > 10%
- Bias < |0.5|

---

## 🎯 Phase 2.5 Success Criteria

**Phase 2.5 is complete when ALL criteria are met:**

| Criterion | Target | Current | Status |
|-----------|--------|---------|--------|
| **Elite + Strong %** | ≥ 10% | 0.25% | ❌ |
| **Overall RMSE** | < 5.0 | 5.707 | ❌ |
| **Filtered RMSE** | < 2.5 | 2.122 | ✅ |
| **Overall Bias** | < \|0.5\| | -1.671 | ❌ |
| **Confidence correlation** | r > 0.5 | Not calculated | ❌ |
| **Phase 2 vs Phase 1** | RMSE improvement | Regression | ❌ |

**Once all criteria are met**, Phase 3 can begin.

---

## 📊 Recommended Task Order

### **Week 1 (Phase 2.5)**:

**Day 1-2**:
- Task 2.5.1: Calculate missing metrics ✅
- Task 2.5.2: Apply bias correction and re-run ✅

**Day 3-4**:
- Task 2.5.3: Feature ablation study ✅
- Task 2.5.4: Model recalibration analysis ✅

**Day 5-7**:
- Task 2.5.5: Adjust confidence thresholds (if needed) ✅
- Task 2.5.6: Model retraining (if needed) ✅
- Final validation backtest ✅

---

## 🚫 Phase 3 Blockers

**DO NOT proceed to Phase 3 until:**

1. ✅ Confidence distribution is usable (≥10% Elite+Strong)
2. ✅ Overall RMSE improves over Phase 1 (< 5.435)
3. ✅ Systematic bias is eliminated (< |0.5|)
4. ✅ Confidence correlation is validated (r > 0.5)
5. ✅ Root cause identified and fixed

**Rationale**: Phase 3 adds complexity (DARKO/EPM, quantile regression, risk management). If the foundation is broken (models disagree 90% of time), adding features will make problems worse.

---

## 💡 Key Learnings

### **What We Learned About the Model**:

1. **Confidence filtering works** - When confident, model is very accurate (RMSE 1.905)
2. **Base models disagree too much** - 90% of predictions have high variance
3. **Phase 2 features may hurt** - Performance regressed after adding features
4. **Systematic bias exists** - Consistently underpredicting by 1.7 units
5. **Implementation is sound** - Code quality is good, results are honest

### **Process Improvements**:

1. **Always calculate confidence metrics** - Don't defer validation
2. **Run ablation studies earlier** - Test features individually before combining
3. **Validate improvements immediately** - Phase 2 should have shown improvement over Phase 1
4. **Monitor confidence distribution** - 90% "avoid" tier is a red flag
5. **Set success criteria upfront** - Clear targets prevent wasted work

---

## 📁 Files to Create (Phase 2.5)

1. `phase2.5_missing_metrics.py` - Calculate correlation and calibration
2. `phase2.5_bias_correction.py` - Apply and validate bias correction
3. `phase2.5_feature_ablation.py` - Test features individually
4. `phase2.5_model_analysis.py` - Analyze base model predictions
5. `backtest_results/phase2.5_results.json` - Final Phase 2.5 metrics
6. `backtest_results/calibration_curve.png` - Visualization
7. `backtest_results/ablation_study.json` - Feature impact analysis

---

## 🎓 Conclusion

Task 2.6 was **technically successful** but revealed **fundamental issues** requiring immediate attention:

✅ **Implementation**: Well-coded, well-documented, follows best practices
❌ **Results**: Model performance declined, confidence distribution unusable
⚠️ **Decision**: Phase 2.5 debugging sprint required before Phase 3

**Bottom Line**: The prediction system shows promise when confident (RMSE 1.905 for Strong tier), but only 0.25% of predictions are confident enough to act on. Phase 2.5 must fix the model calibration and confidence distribution to make this a viable betting system.

---

**Next Action**: Begin Phase 2.5 Task 2.5.1 (Calculate Missing Metrics)
