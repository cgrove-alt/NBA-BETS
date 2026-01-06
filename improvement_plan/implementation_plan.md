# NBA Model V2: "The Oracle" Improvement Plan

## Executive Summary
This document outlines the architectural upgrades required to transform the current NBA Betting Model into the most accurate prediction engine in existence. While the current model excels at statistical regression and hygiene (temporal discipline), it lacks the **dynamic simulation** and **market microstructure** capabilities of top-tier syndicate models.

**Objective**: Move from "Regression-based" to "Simulation & Market-based".

## 1. Simulation Engine ("Possession-Level Modeling")

### Current Limitation:
The current model predicts final scores using regression (XGBoost/MLP). This fails to capture:
- Nonlinearities (e.g., standard deviation of outcomes differs by pace).
- Correlation between player props (e.g., if a game is a blowout, starters play less).
- "Game Script" dependent outcomes.

### Proposed Solution: Monte Carlo Game Simulator
**File**: `simulation_engine.py` (New)

#### Architecture:
1.  **State Machine**: Model the game as a sequence of possessions.
    -   *States*: `StartPossession` -> `ShotAttempt` -> `Rebound/Turnover/MadeShot` -> `Transition`
2.  **Transition Probabilities**:
    -   Derive `P(Shot | Player, Defense)` from player tracking data proxies (or advanced box scores).
    -   Derive `P(Pace)` from matchup dynamics.
3.  **Execution**:
    -   Simulate 10,000 games per matchup.
    -   **Output**: Distribution of scores, player stats, and play-by-play events.

#### Benefit:
-   **True Prop Probabilities**: calculate "Probability LeBron > 25.5 pts" directly from the simulation distribution, which naturally accounts for blowout risk and pace.
-   **Correlated Parlays**: Determine the exact correlation between "Lakers Win" and "LeBron Over", enabling "Same Game Parlay" edge finding.

## 2. Market Microstructure ("Steam & Decay")

### Current Limitation:
The model calculates "Line Movement" based on Open vs Current. This is too coarse. It misses the *timing* and *velocity* of moves.

### Proposed Solution: Real-Time Odds Monitor
**File**: `market_microstructure.py` (New)

#### Features:
1.  **Steam Chaser**:
    -   Poll `The Odds API` every 60s.
    -   Identify "Leader" books (e.g., Pinnacle, Circa) moving first.
    -   Trigger alerts/bets at "Laggard" books (e.g., DraftKings, FanDuel) that haven't moved yet.
2.  **Stale Line Detection**:
    -   Calculate consensus "Fair Odds" excluding the vig.
    -   Identify specific books offering +EV against the consensus.

## 3. Qualitative Intelligence ("The News Edge")

### Current Limitation:
Impact of injuries is binary (`status="out"`) or formulaic (`value_lost`). It misses context (e.g., "Player is active but playing through flu").

### Proposed Solution: LLM Sentiment Agent
**File**: `news_sentiment.py` (New)

#### Architecture:
1.  **Ingestion**: Scrape beat writer tweets and official reports.
2.  **Analysis**: Use Claude to parse text for:
    -   *Severity*: 1-10 scale.
    -   *Context*: "Minutes limit", "Coming off bench".
3.  **Adjustment**: Apply `SentimentPenalty` to player projections before simulation.

## 4. Portfolio Optimization ("Multivariate Kelly")

### Current Limitation:
Bets are sized independently. If we bet "Lakers ML" and "LeBron Over", we are over-exposed to the Lakers performing well.

### Proposed Solution: Covariance-Aware Staking
**File**: `portfolio_optimizer.py` (New)

#### Algorithm:
1.  Construct `CovarianceMatrix` of all pending bets using historical correlation of similar bets.
2.  Optimize allocation to maximize `Geometric Growth` (Kelly) subject to `Drawdown Constraint`.
3.  **Result**: Smaller bet sizes on highly correlated outcomes, larger sizes on uncorrelated edges.

## Action Plan for Claude Code
1.  **Read** this plan.
2.  **Execute** Phase 1 (Simulation Engine) first. This is the biggest accuracy lift.
3.  **Execute** Phase 2 (Market Data) second. This is the biggest ROI lift.
4.  **Mark** tasks in `tasks.md` as you go.
