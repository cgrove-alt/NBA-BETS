export function ConfidenceExplanation() {
  return (
    <div className="space-y-2">
      <div className="font-semibold text-text-primary mb-2">
        Confidence Score (50-85%)
      </div>
      <div className="text-xs text-text-secondary mb-2">
        Data-driven 9-factor calculation based on 6,076 historical predictions:
      </div>

      <div className="space-y-1.5">
        <div className="flex items-start gap-2">
          <span className="text-accent-primary shrink-0">1.</span>
          <div>
            <span className="font-medium text-text-primary">Sample Size Boost</span>
            <div className="text-text-muted">More games played = higher confidence</div>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <span className="text-accent-primary shrink-0">2.</span>
          <div>
            <span className="font-medium text-text-primary">Form Stability</span>
            <div className="text-text-muted">Recent performance vs season average</div>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <span className="text-accent-primary shrink-0">3.</span>
          <div>
            <span className="font-medium text-text-primary">Consistency Score</span>
            <div className="text-text-muted">Low variance = predictable player</div>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <span className="text-accent-primary shrink-0">4.</span>
          <div>
            <span className="font-medium text-text-primary">Edge Magnitude</span>
            <div className="text-text-muted">Sweet spot: 5-15% edge</div>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <span className="text-accent-primary shrink-0">5.</span>
          <div>
            <span className="font-medium text-text-primary">Real Line Available</span>
            <div className="text-text-muted">Sportsbook line exists (+3%)</div>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <span className="text-accent-primary shrink-0">6.</span>
          <div>
            <span className="font-medium text-text-primary">Whitelist Bonus</span>
            <div className="text-text-muted">15 historically accurate players (+10%)</div>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <span className="text-accent-primary shrink-0">7.</span>
          <div>
            <span className="font-medium text-text-primary">Minutes Stability</span>
            <div className="text-text-muted">Consistent playing time variance</div>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <span className="text-accent-primary shrink-0">8.</span>
          <div>
            <span className="font-medium text-text-primary">Matchup Quality</span>
            <div className="text-text-muted">Easy matchup vs weak defense (+5%)</div>
          </div>
        </div>

        <div className="flex items-start gap-2">
          <span className="text-accent-primary shrink-0">9.</span>
          <div>
            <span className="font-medium text-text-primary">Hit Rate Boost</span>
            <div className="text-text-muted">Player beats line {'>'}60% in last 10 games (+5%)</div>
          </div>
        </div>
      </div>

      <div className="mt-3 pt-2 border-t border-border text-xs text-text-muted">
        Distribution: 50-55% (40%), 55-60% (35%), 60-65% (18%), 65-70% (5%), 70-85% (2%)
      </div>
    </div>
  );
}
