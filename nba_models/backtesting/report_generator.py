"""
HTML Backtesting Report Generator with Plotly Visualizations

Generates professional, interactive HTML reports from backtesting results.
Includes executive summary, performance metrics, calibration plots, and worst misses.

Usage:
    python report_generator.py backtest_results/phase3_backtest_2seasons.json

Author: NBA Prediction Model Team
Version: 1.0
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import math

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from jinja2 import Environment, BaseLoader


def load_backtest_results(file_path: str) -> dict[str, Any]:
    """Load backtest results from JSON file."""
    with open(file_path) as f:
        return json.load(f)


def safe_get(data: dict, key: str, default: Any = "N/A") -> Any:
    """Safely get value from dict, handling NaN and None."""
    value = data.get(key, default)

    # Handle NaN
    if isinstance(value, (int, float)) and math.isnan(value):
        return default

    # Handle None
    if value is None:
        return default

    return value


def create_roi_curve(betting_data: dict[str, Any]) -> go.Figure:
    """
    Create cumulative ROI curve over time.

    Note: This requires bet-by-bet history. If not available,
    shows final ROI as a bar chart.
    """
    # Check if we have bet history
    if 'bet_history' in betting_data:
        # Time series plot
        history = betting_data['bet_history']
        dates = [bet['date'] for bet in history]
        cumulative_roi = []
        running_profit = 0
        running_wagered = 0

        for bet in history:
            running_profit += bet.get('profit', 0)
            running_wagered += bet.get('amount', 0)
            roi = (running_profit / running_wagered * 100) if running_wagered > 0 else 0
            cumulative_roi.append(roi)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates,
            y=cumulative_roi,
            mode='lines',
            name='Cumulative ROI',
            line={'color': 'green', 'width': 2},
            fill='tozeroy'
        ))

        fig.update_layout(
            title='ROI Over Time',
            xaxis_title='Date',
            yaxis_title='ROI (%)',
            hovermode='x unified',
            height=400
        )
    else:
        # Simple bar chart with final ROI
        roi = safe_get(betting_data, 'roi', 0)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=['Final ROI'],
            y=[roi],
            marker_color='green' if roi > 0 else 'red',
            text=[f'{roi:.2f}%'],
            textposition='outside'
        ))

        fig.update_layout(
            title='Final ROI',
            yaxis_title='ROI (%)',
            height=400,
            showlegend=False
        )

    return fig


def create_calibration_plot(predictions: list[dict[str, Any]]) -> go.Figure:
    """
    Create calibration plot showing predicted vs actual accuracy.

    Bins predictions by confidence level and shows if model is calibrated.
    """
    # Bin predictions by confidence
    list(range(0, 101, 10))  # 0-10, 10-20, ..., 90-100
    [f'{i}-{i+10}' for i in range(0, 100, 10)]

    bin_counts = [0] * 10
    bin_correct = [0] * 10

    for pred in predictions:
        confidence = pred.get('confidence', 50)
        error = abs(pred.get('error', 0))

        # Determine bin
        bin_idx = min(int(confidence // 10), 9)
        bin_counts[bin_idx] += 1

        # Consider "correct" if error < 5 for props, < 3 for spreads
        threshold = 5.0
        if error < threshold:
            bin_correct[bin_idx] += 1

    # Calculate actual accuracy per bin
    actual_accuracy = []
    for i in range(10):
        if bin_counts[i] > 0:
            actual_accuracy.append(bin_correct[i] / bin_counts[i] * 100)
        else:
            actual_accuracy.append(None)

    # Expected accuracy (midpoint of bin)
    expected_accuracy = [i * 10 + 5 for i in range(10)]

    fig = go.Figure()

    # Perfect calibration line
    fig.add_trace(go.Scatter(
        x=[0, 100],
        y=[0, 100],
        mode='lines',
        name='Perfect Calibration',
        line={'color': 'gray', 'dash': 'dash', 'width': 2}
    ))

    # Actual calibration
    fig.add_trace(go.Scatter(
        x=expected_accuracy,
        y=actual_accuracy,
        mode='markers+lines',
        name='Actual Calibration',
        marker={'size': 10, 'color': 'blue'},
        line={'color': 'blue', 'width': 2}
    ))

    fig.update_layout(
        title='Calibration Plot: Predicted vs Actual Accuracy',
        xaxis_title='Predicted Confidence (%)',
        yaxis_title='Actual Accuracy (%)',
        hovermode='x unified',
        height=400
    )

    return fig


def create_tier_performance_chart(tier_data: dict[str, dict[str, Any]]) -> go.Figure:
    """Create bar chart comparing performance across confidence tiers."""
    tiers = []
    rmse_values = []
    counts = []

    tier_order = ['elite', 'strong', 'moderate', 'weak', 'avoid']

    for tier in tier_order:
        if tier in tier_data:
            data = tier_data[tier]
            tiers.append(tier.capitalize())
            rmse_values.append(safe_get(data, 'rmse', 0))
            counts.append(safe_get(data, 'count', 0))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RMSE by Tier', 'Prediction Count by Tier'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # RMSE chart
    fig.add_trace(
        go.Bar(
            x=tiers,
            y=rmse_values,
            name='RMSE',
            marker_color=['gold', 'green', 'orange', 'red', 'darkred'][:len(tiers)],
            text=[f'{v:.2f}' for v in rmse_values],
            textposition='outside'
        ),
        row=1, col=1
    )

    # Count chart
    fig.add_trace(
        go.Bar(
            x=tiers,
            y=counts,
            name='Count',
            marker_color=['gold', 'green', 'orange', 'red', 'darkred'][:len(tiers)],
            text=counts,
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        showlegend=False
    )

    return fig


def create_prop_type_comparison(prop_data: dict[str, dict[str, Any]]) -> go.Figure:
    """Create comparison of performance across different prop types."""
    prop_types = []
    rmse_values = []
    r2_values = []

    for prop_type, data in prop_data.items():
        prop_types.append(prop_type.upper())
        rmse_values.append(safe_get(data, 'rmse', 0))
        r2_values.append(safe_get(data, 'r2', 0))

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('RMSE by Prop Type', 'R² by Prop Type'),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )

    # RMSE chart
    fig.add_trace(
        go.Bar(
            x=prop_types,
            y=rmse_values,
            name='RMSE',
            marker_color='steelblue',
            text=[f'{v:.2f}' for v in rmse_values],
            textposition='outside'
        ),
        row=1, col=1
    )

    # R² chart (color by positive/negative)
    colors = ['green' if r2 > 0 else 'red' for r2 in r2_values]
    fig.add_trace(
        go.Bar(
            x=prop_types,
            y=r2_values,
            name='R²',
            marker_color=colors,
            text=[f'{v:.3f}' for v in r2_values],
            textposition='outside'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=400,
        showlegend=False
    )

    return fig


def create_worst_misses_table(predictions: list[dict[str, Any]], top_n: int = 20) -> str:
    """Create HTML table of worst prediction misses."""
    # Sort by absolute error
    sorted_preds = sorted(predictions, key=lambda x: abs(x.get('error', 0)), reverse=True)
    worst = sorted_preds[:top_n]

    html = """
    <table class="table table-striped table-hover">
        <thead class="table-dark">
            <tr>
                <th>Rank</th>
                <th>Player</th>
                <th>Prop Type</th>
                <th>Predicted</th>
                <th>Actual</th>
                <th>Error</th>
                <th>Confidence</th>
                <th>Tier</th>
                <th>Date</th>
            </tr>
        </thead>
        <tbody>
    """

    for i, pred in enumerate(worst, 1):
        error = pred.get('error', 0)
        error_class = 'text-danger' if abs(error) > 10 else 'text-warning'

        html += f"""
            <tr>
                <td>{i}</td>
                <td>{safe_get(pred, 'player', 'Unknown')}</td>
                <td>{safe_get(pred, 'prop_type', 'N/A').upper()}</td>
                <td>{safe_get(pred, 'predicted', 0):.2f}</td>
                <td>{safe_get(pred, 'actual', 0):.1f}</td>
                <td class="{error_class}"><strong>{error:.2f}</strong></td>
                <td>{safe_get(pred, 'confidence', 0):.0f}%</td>
                <td>{safe_get(pred, 'tier', 'N/A').upper()}</td>
                <td>{safe_get(pred, 'game_date', 'N/A')}</td>
            </tr>
        """

    html += """
        </tbody>
    </table>
    """

    return html


def generate_html_report(backtest_file: str, output_path: str | None = None) -> str:
    """
    Generate comprehensive HTML backtest report with Plotly visualizations.

    Args:
        backtest_file: Path to backtest results JSON file
        output_path: Optional output path for HTML file (auto-generated if None)

    Returns:
        Path to generated HTML report
    """
    # Load data
    results = load_backtest_results(backtest_file)

    # Determine which season to report on
    # Prefer season with actual data
    season_key = None
    season_data = None

    if 'season_2025_26' in results and 'error' not in results['season_2025_26']:
        season_key = 'season_2025_26'
        season_data = results['season_2025_26']
    elif 'season_2024_25' in results and 'error' not in results['season_2024_25']:
        season_key = 'season_2024_25'
        season_data = results['season_2024_25']
    else:
        # Single season result
        season_data = results
        season_key = 'combined'

    # Extract data sections
    overall = season_data.get('overall_performance', {})
    tier_perf = season_data.get('tier_performance', {})
    prop_perf = season_data.get('prop_type_performance', {})
    betting = season_data.get('betting_performance', {})
    calibration = season_data.get('calibration', {})
    elite_strong = season_data.get('elite_strong_performance', {})
    sample_preds = season_data.get('sample_predictions', [])

    # Generate Plotly charts
    roi_chart = create_roi_curve(betting)
    calibration_chart = create_calibration_plot(sample_preds)
    tier_chart = create_tier_performance_chart(tier_perf)
    prop_chart = create_prop_type_comparison(prop_perf)
    worst_table = create_worst_misses_table(sample_preds, top_n=20)

    # Convert charts to HTML
    roi_html = roi_chart.to_html(full_html=False, include_plotlyjs='cdn')
    calibration_html = calibration_chart.to_html(full_html=False, include_plotlyjs=False)
    tier_html = tier_chart.to_html(full_html=False, include_plotlyjs=False)
    prop_html = prop_chart.to_html(full_html=False, include_plotlyjs=False)

    # Custom Jinja2 filter for number formatting
    def number_format(value):
        """Format number with commas."""
        try:
            return f"{float(value):,.2f}"
        except:
            return str(value)

    # Create Jinja2 environment with custom filter
    env = Environment(loader=BaseLoader())
    env.filters['number_format'] = number_format

    # HTML Template
    template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NBA Prediction Model - Backtest Report</title>

    <!-- Bootstrap CSS -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">

    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f8f9fa;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        .metric-label {
            color: #6c757d;
            font-size: 0.9rem;
            text-transform: uppercase;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
        .section-title {
            margin: 2rem 0 1rem 0;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #667eea;
        }
        .chart-container {
            background: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .target-status {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 4px;
            font-weight: bold;
            font-size: 0.85rem;
        }
        .target-met {
            background-color: #d4edda;
            color: #155724;
        }
        .target-missed {
            background-color: #f8d7da;
            color: #721c24;
        }
        .footer {
            margin-top: 3rem;
            padding: 2rem;
            background: #343a40;
            color: white;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="container">
            <h1>🏀 NBA Prediction Model - Backtest Report</h1>
            <p class="lead">{{ phase }} - Completed {{ date_completed }}</p>
            <p>Season: <strong>{{ season_key }}</strong> | Total Predictions: <strong>{{ total_predictions | number_format }}</strong></p>
        </div>
    </div>

    <div class="container">
        <!-- Executive Summary -->
        <h2 class="section-title">📊 Executive Summary</h2>
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Total ROI</div>
                    <div class="metric-value {{ 'positive' if betting_roi_raw > 0 else 'negative' }}">
                        {{ betting_roi }}%
                    </div>
                    <small>Target: > 3%</small>
                    <span class="target-status {{ 'target-met' if betting_roi_raw > 3 else 'target-missed' }}">
                        {{ '✓ MET' if betting_roi_raw > 3 else '✗ MISSED' }}
                    </span>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Win Rate</div>
                    <div class="metric-value {{ 'positive' if win_rate_raw > 52 else 'negative' }}">
                        {{ win_rate }}%
                    </div>
                    <small>Target: 52-58%</small>
                    <span class="target-status {{ 'target-met' if win_rate_raw >= 52 and win_rate_raw <= 58 else 'target-missed' }}">
                        {{ '✓ MET' if win_rate_raw >= 52 and win_rate_raw <= 58 else '✗ MISSED' }}
                    </span>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value {{ 'positive' if sharpe_raw > 1.5 else 'negative' }}">
                        {{ sharpe }}
                    </div>
                    <small>Target: > 1.5</small>
                    <span class="target-status {{ 'target-met' if sharpe_raw > 1.5 else 'target-missed' }}">
                        {{ '✓ MET' if sharpe_raw > 1.5 else '✗ MISSED' }}
                    </span>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value {{ 'positive' if max_drawdown_raw < 15 else 'negative' }}">
                        {{ max_drawdown }}%
                    </div>
                    <small>Target: < 15%</small>
                    <span class="target-status {{ 'target-met' if max_drawdown_raw < 15 else 'target-missed' }}">
                        {{ '✓ MET' if max_drawdown_raw < 15 else '✗ MISSED' }}
                    </span>
                </div>
            </div>
        </div>

        <!-- Overall Performance -->
        <h2 class="section-title">📈 Overall Performance</h2>
        <div class="row">
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value">{{ overall_rmse }}</div>
                    <small>Target: < 4.8</small>
                    <span class="target-status {{ 'target-met' if overall_rmse_raw < 4.8 else 'target-missed' }}">
                        {{ '✓ MET' if overall_rmse_raw < 4.8 else '✗ MISSED' }}
                    </span>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">MAE</div>
                    <div class="metric-value">{{ overall_mae }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Bias</div>
                    <div class="metric-value">{{ overall_bias }}</div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="metric-card">
                    <div class="metric-label">Elite+Strong %</div>
                    <div class="metric-value positive">{{ elite_strong_pct }}%</div>
                    <small>RMSE: {{ elite_strong_rmse }}</small>
                </div>
            </div>
        </div>

        <!-- ROI Chart -->
        <h2 class="section-title">💰 ROI Performance</h2>
        <div class="chart-container">
            {{ roi_chart | safe }}
        </div>

        <div class="row">
            <div class="col-md-4">
                <div class="metric-card">
                    <div class="metric-label">Total Bets</div>
                    <div class="metric-value">{{ total_bets }}</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <div class="metric-label">Total Wagered</div>
                    <div class="metric-value">${{ total_wagered | number_format }}</div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="metric-card">
                    <div class="metric-label">Total Profit</div>
                    <div class="metric-value {{ 'positive' if total_profit_raw > 0 else 'negative' }}">
                        ${{ total_profit | number_format }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Performance by Tier -->
        <h2 class="section-title">🎯 Performance by Confidence Tier</h2>
        <div class="chart-container">
            {{ tier_chart | safe }}
        </div>

        <!-- Performance by Prop Type -->
        <h2 class="section-title">🏆 Performance by Prop Type</h2>
        <div class="chart-container">
            {{ prop_chart | safe }}
        </div>

        <!-- Calibration Analysis -->
        <h2 class="section-title">📏 Calibration Analysis</h2>
        <div class="chart-container">
            {{ calibration_chart | safe }}
        </div>

        <div class="row">
            <div class="col-md-6">
                <div class="metric-card">
                    <div class="metric-label">Confidence-Accuracy Correlation</div>
                    <div class="metric-value {{ 'positive' if conf_corr_raw > 0.5 else 'negative' }}">
                        {{ conf_corr }}
                    </div>
                    <small>Target: > 0.5</small>
                    <span class="target-status {{ 'target-met' if conf_corr_raw > 0.5 else 'target-missed' }}">
                        {{ '✓ MET' if conf_corr_raw > 0.5 else '✗ MISSED' }}
                    </span>
                </div>
            </div>
            <div class="col-md-6">
                <div class="metric-card">
                    <div class="metric-label">Average Confidence (All Predictions)</div>
                    <div class="metric-value">{{ avg_confidence }}%</div>
                </div>
            </div>
        </div>

        <!-- Worst Misses -->
        <h2 class="section-title">⚠️ Worst Prediction Misses (Top 20)</h2>
        <div class="metric-card">
            {{ worst_misses_table | safe }}
        </div>

        <!-- Key Insights -->
        <h2 class="section-title">💡 Key Insights</h2>
        <div class="metric-card">
            <ul>
                <li><strong>Model Status:</strong> {{ model_status }}</li>
                <li><strong>Best Performing Prop:</strong> {{ best_prop }} (R² = {{ best_prop_r2 }})</li>
                <li><strong>Worst Performing Prop:</strong> {{ worst_prop }} (R² = {{ worst_prop_r2 }})</li>
                <li><strong>Elite+Strong Tier:</strong> {{ elite_strong_pct }}% of predictions with RMSE {{ elite_strong_rmse }}</li>
                <li><strong>Betting Strategy:</strong> {{ betting_strategy }}</li>
            </ul>
        </div>

        <!-- Recommendations -->
        <h2 class="section-title">🚀 Recommendations</h2>
        <div class="metric-card">
            <ul>
                {{ recommendations | safe }}
            </ul>
        </div>
    </div>

    <div class="footer">
        <p>Generated on {{ generation_date }}</p>
        <p>NBA Prediction Model v2.0 - Phase 3 Optimization</p>
    </div>

    <!-- Bootstrap JS -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

    # Generate recommendations
    recommendations = []

    if safe_get(betting, 'roi', 0) > 3:
        recommendations.append('<li class="positive">✓ <strong>APPROVED for paper trading</strong> - ROI exceeds 3% target</li>')
    else:
        recommendations.append('<li class="negative">✗ <strong>NOT APPROVED</strong> - ROI below 3% target, requires retraining</li>')

    if safe_get(elite_strong, 'rmse', 10) < 4.8:
        recommendations.append('<li class="positive">✓ Focus on Elite+Strong tier bets for optimal performance</li>')

    # Identify worst prop
    worst_prop = None
    worst_r2 = 1.0
    for prop, data in prop_perf.items():
        r2 = safe_get(data, 'r2', 0)
        if r2 < worst_r2:
            worst_r2 = r2
            worst_prop = prop

    if worst_prop and worst_r2 < 0:
        recommendations.append(f'<li class="negative">✗ Avoid {worst_prop.upper()} props entirely - negative R² indicates unpredictability</li>')

    # Best prop
    best_prop = None
    best_r2 = -1.0
    for prop, data in prop_perf.items():
        r2 = safe_get(data, 'r2', -1)
        if r2 > best_r2:
            best_r2 = r2
            best_prop = prop

    if best_prop and best_r2 > 0:
        recommendations.append(f'<li class="positive">✓ Prioritize {best_prop.upper()} props - best R² score</li>')

    if safe_get(calibration, 'confidence_accuracy_correlation', 0) > 0.5:
        recommendations.append('<li class="positive">✓ Confidence scoring is well-calibrated - trust tier recommendations</li>')
    else:
        recommendations.append('<li class="negative">✗ Recalibrate confidence scoring - correlation below target</li>')

    # Determine model status
    roi = safe_get(betting, 'roi', 0)
    if roi > 7:
        model_status = "🟢 EXCELLENT - Exceeding all targets"
    elif roi > 3:
        model_status = "🟡 GOOD - Meeting profitability targets"
    elif roi > 0:
        model_status = "🟠 MARGINAL - Profitable but below target"
    else:
        model_status = "🔴 POOR - Not profitable, requires retraining"

    # Betting strategy
    if roi > 5:
        betting_strategy = "Approved for 25% bankroll allocation with 1/4 Kelly sizing"
    elif roi > 3:
        betting_strategy = "Approved for 10% bankroll allocation with 1/4 Kelly sizing (paper trading)"
    else:
        betting_strategy = "NOT APPROVED - Continue development and backtesting"

    # Create template from string
    template = env.from_string(template_str)

    # Extract raw values for comparisons
    betting_roi_raw = safe_get(betting, 'roi', 0)
    win_rate_raw = safe_get(betting, 'win_rate', 0)
    sharpe_raw = safe_get(betting, 'sharpe_ratio', 0)
    max_drawdown_raw = safe_get(betting, 'max_drawdown', 0)
    overall_rmse_raw = safe_get(overall, 'rmse', 0)
    conf_corr_raw = safe_get(calibration, 'confidence_accuracy_correlation', 0)
    elite_strong_rmse_raw = safe_get(elite_strong, 'rmse', 0)
    total_profit_raw = safe_get(betting, 'total_profit', 0)

    # Render template
    html_output = template.render(
        # Metadata
        phase=safe_get(season_data, 'phase', 'Unknown Phase'),
        date_completed=safe_get(season_data, 'date_completed', datetime.now().strftime('%Y-%m-%d')),
        season_key=season_key.replace('_', ' ').title(),
        total_predictions=safe_get(season_data, 'total_predictions', 0),
        generation_date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),

        # Executive metrics (formatted for display, raw for comparisons)
        betting_roi=f"{betting_roi_raw:.2f}",
        betting_roi_raw=betting_roi_raw,
        win_rate=f"{win_rate_raw:.2f}",
        win_rate_raw=win_rate_raw,
        sharpe=f"{sharpe_raw:.2f}",
        sharpe_raw=sharpe_raw,
        max_drawdown=f"{max_drawdown_raw:.2f}",
        max_drawdown_raw=max_drawdown_raw,

        # Overall performance
        overall_rmse=f"{overall_rmse_raw:.2f}",
        overall_rmse_raw=overall_rmse_raw,
        overall_mae=f"{safe_get(overall, 'mae', 0):.2f}",
        overall_bias=f"{safe_get(overall, 'bias', 0):.2f}",
        elite_strong_pct=f"{safe_get(elite_strong, 'percentage', 0):.1f}",
        elite_strong_rmse=f"{elite_strong_rmse_raw:.2f}",

        # Betting details
        total_bets=safe_get(betting, 'total_bets', 0),
        total_wagered=f"{safe_get(betting, 'total_wagered', 0):.2f}",
        total_profit=f"{total_profit_raw:.2f}",
        total_profit_raw=total_profit_raw,

        # Calibration
        conf_corr=f"{conf_corr_raw:.3f}",
        conf_corr_raw=conf_corr_raw,
        avg_confidence=f"{safe_get(calibration, 'avg_confidence_all', 0):.1f}",

        # Charts
        roi_chart=roi_html,
        calibration_chart=calibration_html,
        tier_chart=tier_html,
        prop_chart=prop_html,
        worst_misses_table=worst_table,

        # Insights
        model_status=model_status,
        best_prop=best_prop.upper() if best_prop else 'N/A',
        best_prop_r2=f"{best_r2:.3f}",
        worst_prop=worst_prop.upper() if worst_prop else 'N/A',
        worst_prop_r2=f"{worst_r2:.3f}",
        betting_strategy=betting_strategy,
        recommendations='\n'.join(recommendations)
    )

    # Determine output path
    if output_path is None:
        backtest_name = Path(backtest_file).stem
        output_path = f"backtest_reports/{backtest_name}_report.html"

    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write output
    with open(output_path, 'w') as f:
        f.write(html_output)

    print("\n✅ Report generated successfully!")
    print(f"📊 Output: {output_path}")
    print(f"📈 Total Predictions: {safe_get(season_data, 'total_predictions', 0)}")
    print(f"💰 ROI: {safe_get(betting, 'roi', 0):.2f}%")
    print(f"🎯 Win Rate: {safe_get(betting, 'win_rate', 0):.2f}%")

    return output_path


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python report_generator.py <backtest_results.json> [output.html]")
        print("\nExample:")
        print("  python report_generator.py backtest_results/phase3_backtest_2seasons.json")
        sys.exit(1)

    backtest_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(backtest_file).exists():
        print(f"❌ Error: File not found: {backtest_file}")
        sys.exit(1)

    try:
        report_path = generate_html_report(backtest_file, output_file)
        print("\n🎉 Success! Open the report in your browser:")
        print(f"   file://{Path(report_path).absolute()}")
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
