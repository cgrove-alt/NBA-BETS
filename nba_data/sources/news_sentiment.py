"""
NBA News Sentiment Analysis

LLM-powered analysis of news, injury reports, and beat writer information
to enhance betting predictions with qualitative intelligence.

=============================================================================
KEY CAPABILITIES
=============================================================================
1. NewsIngestor: Collect and process news from multiple sources
2. SentimentAnalyzer: Use Claude to extract actionable insights
3. InjuryImpactCalculator: Convert qualitative info to quantitative adjustments
4. IntegrationPipeline: Feed insights into prediction models

INSIGHT TYPES:
- Injury Context: "Playing through flu", "minutes restriction expected"
- Motivation: "Contract year", "revenge game", "playoff push"
- Team Dynamics: "Locker room issues", "chemistry problems"
- Rest/Load: "Back-to-back concerns", "fresh legs"
- Matchup Notes: "Career-high against this opponent", "struggles vs zone"
=============================================================================
"""

import load_env  # noqa: F401  — load .env before any code reads os.environ
import os
import time
import json
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import threading


# =============================================================================
# CONFIGURATION
# =============================================================================

# Cache settings
CACHE_DIR = Path(__file__).parent / ".news_cache"
NEWS_CACHE_TTL = 1800  # 30 minutes

# Severity mapping for injury context
SEVERITY_ADJUSTMENTS = {
    1: 0.0,   # Negligible - no adjustment
    2: -0.02,  # Minor - 2% reduction
    3: -0.05,  # Moderate - 5% reduction
    4: -0.10,  # Significant - 10% reduction
    5: -0.15,  # Major - 15% reduction
    6: -0.20,  # Severe - 20% reduction
    7: -0.30,  # Very Severe - 30% reduction
    8: -0.40,  # Critical - 40% reduction
    9: -0.50,  # Near-Out - 50% reduction
    10: -0.75,  # Essentially Out - 75% reduction
}


class InsightType(Enum):
    """Types of news insights."""
    INJURY = "injury"
    MOTIVATION = "motivation"
    TEAM_DYNAMICS = "team_dynamics"
    REST_LOAD = "rest_load"
    MATCHUP = "matchup"
    LINEUP = "lineup"
    TRADE = "trade"
    GENERAL = "general"


@dataclass
class NewsItem:
    """A single news item."""
    id: str
    source: str
    timestamp: float
    headline: str
    content: str
    url: str | None = None

    # Extracted entities
    players: list[str] = field(default_factory=list)
    teams: list[str] = field(default_factory=list)

    # Analysis results
    analyzed: bool = False
    insights: list[dict] = field(default_factory=list)


@dataclass
class PlayerInsight:
    """Insight about a specific player."""
    player_name: str
    player_id: int | None = None
    team: str | None = None

    insight_type: InsightType = InsightType.GENERAL
    severity: int = 1  # 1-10 scale
    context: str = ""  # Human-readable context
    confidence: float = 0.5  # 0-1 confidence in insight

    # Quantitative adjustments
    points_adjustment: float = 0.0  # Multiplier (e.g., 0.9 = 10% reduction)
    minutes_adjustment: float = 0.0  # Expected minutes change
    usage_adjustment: float = 0.0  # Usage rate adjustment

    # Metadata
    source_news_id: str = ""
    timestamp: float = 0.0
    expires_at: float = 0.0  # When insight becomes stale


@dataclass
class TeamInsight:
    """Insight about a team."""
    team_name: str
    team_id: int | None = None

    insight_type: InsightType = InsightType.GENERAL
    impact_score: float = 0.0  # -1 to +1 (-1 = very negative, +1 = very positive)
    context: str = ""
    confidence: float = 0.5

    # Affected markets
    affects_moneyline: bool = True
    affects_spread: bool = True
    affects_total: bool = False

    source_news_id: str = ""
    timestamp: float = 0.0


# =============================================================================
# NEWS INGESTOR
# =============================================================================

class NewsIngestor:
    """
    Collects and normalizes news from multiple sources.

    Supports:
    - Manual news input
    - RSS feeds (future)
    - Twitter/X API (future)
    - ESPN/NBA.com scraping (future)
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

        self.news_items: dict[str, NewsItem] = {}
        self._lock = threading.Lock()

        # Load cached news
        self._load_cache()

    def add_news(
        self,
        headline: str,
        content: str,
        source: str = "manual",
        url: str = None,
        timestamp: float = None
    ) -> NewsItem:
        """
        Add a news item for analysis.

        Args:
            headline: News headline
            content: Full content/body
            source: Source name (e.g., "twitter", "espn", "manual")
            url: Source URL (optional)
            timestamp: Unix timestamp (defaults to now)

        Returns:
            NewsItem object
        """
        news_id = hashlib.md5(f"{headline}_{content[:100]}".encode()).hexdigest()[:12]

        item = NewsItem(
            id=news_id,
            source=source,
            timestamp=timestamp or time.time(),
            headline=headline,
            content=content,
            url=url,
        )

        # Extract entities
        item.players = self._extract_players(headline + " " + content)
        item.teams = self._extract_teams(headline + " " + content)

        with self._lock:
            self.news_items[news_id] = item
            self._save_cache()

        return item

    def get_news_for_player(self, player_name: str, hours: int = 24) -> list[NewsItem]:
        """Get recent news mentioning a player."""
        cutoff = time.time() - (hours * 3600)
        name_lower = player_name.lower()

        results = []
        with self._lock:
            for item in self.news_items.values():
                if item.timestamp < cutoff:
                    continue
                # Check if player mentioned
                if name_lower in item.headline.lower() or name_lower in item.content.lower() or any(name_lower in p.lower() for p in item.players):
                    results.append(item)

        return sorted(results, key=lambda x: x.timestamp, reverse=True)

    def get_news_for_team(self, team_name: str, hours: int = 24) -> list[NewsItem]:
        """Get recent news mentioning a team."""
        cutoff = time.time() - (hours * 3600)
        team_lower = team_name.lower()

        results = []
        with self._lock:
            for item in self.news_items.values():
                if item.timestamp < cutoff:
                    continue
                if team_lower in item.headline.lower() or team_lower in item.content.lower() or any(team_lower in t.lower() for t in item.teams):
                    results.append(item)

        return sorted(results, key=lambda x: x.timestamp, reverse=True)

    def get_unanalyzed_news(self) -> list[NewsItem]:
        """Get news items that haven't been analyzed yet."""
        with self._lock:
            return [item for item in self.news_items.values() if not item.analyzed]

    def mark_analyzed(self, news_id: str, insights: list[dict]):
        """Mark a news item as analyzed with insights."""
        with self._lock:
            if news_id in self.news_items:
                self.news_items[news_id].analyzed = True
                self.news_items[news_id].insights = insights
                self._save_cache()

    def _extract_players(self, text: str) -> list[str]:
        """Extract player names from text (basic pattern matching)."""
        # This is a simplified extraction - in production, would use NER
        # Pattern: Capitalized first name + Capitalized last name
        pattern = r'\b([A-Z][a-z]+)\s+([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\b'
        matches = re.findall(pattern, text)

        # Filter common false positives
        stopwords = {'The', 'This', 'That', 'When', 'What', 'Where', 'Which', 'How',
                    'Los', 'New', 'San', 'Golden', 'Oklahoma', 'Portland', 'Miami',
                    'Boston', 'Chicago', 'Denver', 'Houston', 'Dallas'}

        players = []
        for first, last in matches:
            if first not in stopwords:
                players.append(f"{first} {last}")

        return list(set(players))

    def _extract_teams(self, text: str) -> list[str]:
        """Extract NBA team names from text."""
        nba_teams = [
            'Lakers', 'Celtics', 'Warriors', 'Nets', 'Knicks', 'Heat', 'Bulls',
            'Suns', 'Mavericks', 'Bucks', 'Sixers', '76ers', 'Clippers', 'Nuggets',
            'Grizzlies', 'Cavaliers', 'Hawks', 'Raptors', 'Timberwolves', 'Pelicans',
            'Kings', 'Blazers', 'Trail Blazers', 'Pacers', 'Hornets', 'Wizards',
            'Magic', 'Pistons', 'Thunder', 'Spurs', 'Jazz', 'Rockets'
        ]

        found = []
        text_lower = text.lower()
        for team in nba_teams:
            if team.lower() in text_lower:
                found.append(team)

        return list(set(found))

    def _load_cache(self):
        """Load cached news from disk."""
        cache_file = self.cache_dir / "news_items.json"
        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    for item_data in data.get('items', []):
                        item = NewsItem(**item_data)
                        self.news_items[item.id] = item
            except (OSError, json.JSONDecodeError, TypeError):
                pass

    def _save_cache(self):
        """Save news to disk cache."""
        cache_file = self.cache_dir / "news_items.json"
        try:
            items_data = [
                {
                    'id': item.id,
                    'source': item.source,
                    'timestamp': item.timestamp,
                    'headline': item.headline,
                    'content': item.content,
                    'url': item.url,
                    'players': item.players,
                    'teams': item.teams,
                    'analyzed': item.analyzed,
                    'insights': item.insights,
                }
                for item in self.news_items.values()
            ]
            with open(cache_file, 'w') as f:
                json.dump({'items': items_data}, f)
        except OSError:
            pass


# =============================================================================
# SENTIMENT ANALYZER (Claude-powered)
# =============================================================================

class SentimentAnalyzer:
    """
    LLM-powered analysis of NBA news for betting insights.

    Uses Claude API to extract:
    - Injury severity and context
    - Motivation factors
    - Team dynamics
    - Lineup implications
    """

    ANALYSIS_PROMPT = """You are an expert NBA analyst helping a sports betting model understand news context.

Analyze the following NBA news item and extract betting-relevant insights.

NEWS:
Headline: {headline}
Content: {content}

For each player or team mentioned, provide:
1. insight_type: One of [injury, motivation, team_dynamics, rest_load, matchup, lineup, trade, general]
2. severity: 1-10 scale where:
   - 1-2: Negligible impact
   - 3-4: Minor impact
   - 5-6: Moderate impact
   - 7-8: Significant impact
   - 9-10: Major/severe impact
3. context: Brief explanation of the insight
4. points_adjustment: Suggested multiplier for points projection (e.g., 0.85 means 15% reduction)
5. minutes_adjustment: Expected minutes change (e.g., -5 means 5 fewer minutes)
6. affects: Which markets are affected [moneyline, spread, total, props]

Respond in JSON format:
{{
  "players": [
    {{
      "name": "Player Name",
      "insight_type": "injury",
      "severity": 6,
      "context": "Expected to play through ankle soreness on minutes restriction",
      "points_adjustment": 0.80,
      "minutes_adjustment": -8,
      "confidence": 0.7
    }}
  ],
  "teams": [
    {{
      "name": "Team Name",
      "insight_type": "rest_load",
      "impact_score": -0.3,
      "context": "Third game in four nights, fatigue likely",
      "affects": ["spread", "total"],
      "confidence": 0.6
    }}
  ],
  "summary": "One-line summary of betting implications"
}}

If no betting-relevant insights, return empty arrays.
Be conservative with severity ratings - only high severity for confirmed major news."""

    def __init__(self, api_key: str = None, model: str = "claude-sonnet-4-20250514"):
        """
        Initialize the sentiment analyzer.

        Args:
            api_key: Anthropic API key (or from ANTHROPIC_API_KEY env var)
            model: Claude model to use
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None

        # Rate limiting
        self.last_call_time = 0
        self.min_call_interval = 1.0  # seconds between calls

        # Cache analyzed results
        self.analysis_cache: dict[str, dict] = {}

    def _init_client(self):
        """Lazy initialization of Anthropic client."""
        if self.client is not None:
            return

        if not self.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable."
            )

        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        except ImportError:
            raise ImportError(
                "anthropic package required. Install with: pip install anthropic"
            )

    def analyze(self, news_item: NewsItem) -> dict:
        """
        Analyze a news item using Claude.

        Args:
            news_item: NewsItem to analyze

        Returns:
            Analysis dictionary with players, teams, and summary
        """
        # Check cache
        cache_key = news_item.id
        if cache_key in self.analysis_cache:
            return self.analysis_cache[cache_key]

        try:
            self._init_client()
        except (ValueError, ImportError):
            # Return fallback analysis if API not available
            return self._fallback_analysis(news_item)

        # Rate limiting
        now = time.time()
        if now - self.last_call_time < self.min_call_interval:
            time.sleep(self.min_call_interval - (now - self.last_call_time))

        try:
            prompt = self.ANALYSIS_PROMPT.format(
                headline=news_item.headline,
                content=news_item.content
            )

            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            self.last_call_time = time.time()

            # Parse JSON response
            result_text = response.content[0].text

            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'\{[\s\S]*\}', result_text)
            if json_match:
                result = json.loads(json_match.group())
            else:
                result = {'players': [], 'teams': [], 'summary': 'No insights extracted'}

            # Cache result
            self.analysis_cache[cache_key] = result

            return result

        except Exception as e:
            print(f"Claude analysis error: {e}")
            return self._fallback_analysis(news_item)

    def _fallback_analysis(self, news_item: NewsItem) -> dict:
        """
        Rule-based fallback when Claude API is unavailable.
        """
        players = []
        teams = []

        text = (news_item.headline + " " + news_item.content).lower()

        # Injury keywords
        injury_keywords = {
            'out': (8, -1.0, -30),  # severity, points_mult, mins_change
            'doubtful': (7, 0.25, -25),
            'questionable': (5, 0.50, -10),
            'probable': (3, 0.90, -5),
            'day-to-day': (5, 0.60, -8),
            'ankle': (4, 0.85, -5),
            'knee': (5, 0.80, -8),
            'back': (5, 0.80, -8),
            'hamstring': (5, 0.75, -10),
            'concussion': (7, 0.30, -20),
            'minutes restriction': (4, 0.75, -10),
            'load management': (3, 0.80, -8),
            'rest': (2, 0.90, -5),
        }

        for keyword, (severity, pts_mult, mins_change) in injury_keywords.items():
            if keyword in text:
                for player in news_item.players:
                    players.append({
                        'name': player,
                        'insight_type': 'injury',
                        'severity': severity,
                        'context': f"Keyword '{keyword}' detected",
                        'points_adjustment': pts_mult,
                        'minutes_adjustment': mins_change,
                        'confidence': 0.4,
                    })
                break  # Only match first keyword

        # Motivation keywords
        motivation_keywords = {
            'revenge game': (0.1, 'motivation'),
            'contract year': (0.05, 'motivation'),
            'return to': (0.05, 'motivation'),
            'playoff': (0.05, 'motivation'),
            'elimination': (0.1, 'motivation'),
        }

        for keyword, (impact, itype) in motivation_keywords.items():
            if keyword in text:
                for team in news_item.teams:
                    teams.append({
                        'name': team,
                        'insight_type': itype,
                        'impact_score': impact,
                        'context': f"Keyword '{keyword}' detected",
                        'affects': ['moneyline', 'spread'],
                        'confidence': 0.3,
                    })

        return {
            'players': players,
            'teams': teams,
            'summary': 'Fallback rule-based analysis (API unavailable)',
        }

    def batch_analyze(self, news_items: list[NewsItem]) -> list[dict]:
        """Analyze multiple news items."""
        return [self.analyze(item) for item in news_items]


# =============================================================================
# INJURY IMPACT CALCULATOR
# =============================================================================

class InjuryImpactCalculator:
    """
    Converts news insights into quantitative adjustments for predictions.

    Integrates with injury_fetcher.py to provide enhanced injury context.
    """

    def __init__(self, news_ingestor: NewsIngestor, analyzer: SentimentAnalyzer):
        self.ingestor = news_ingestor
        self.analyzer = analyzer

        # Store calculated impacts
        self.player_impacts: dict[str, PlayerInsight] = {}
        self.team_impacts: dict[str, TeamInsight] = {}

    def process_all_news(self):
        """Process all unanalyzed news and update impacts."""
        unanalyzed = self.ingestor.get_unanalyzed_news()

        for news_item in unanalyzed:
            analysis = self.analyzer.analyze(news_item)

            # Extract player insights
            for player_data in analysis.get('players', []):
                insight = self._create_player_insight(player_data, news_item)
                if insight:
                    # Update if more severe or newer
                    existing = self.player_impacts.get(insight.player_name)
                    if not existing or insight.severity > existing.severity:
                        self.player_impacts[insight.player_name] = insight

            # Extract team insights
            for team_data in analysis.get('teams', []):
                insight = self._create_team_insight(team_data, news_item)
                if insight:
                    existing = self.team_impacts.get(insight.team_name)
                    if not existing or abs(insight.impact_score) > abs(existing.impact_score):
                        self.team_impacts[insight.team_name] = insight

            # Mark as analyzed
            self.ingestor.mark_analyzed(news_item.id, analysis.get('players', []) + analysis.get('teams', []))

    def _create_player_insight(self, data: dict, news_item: NewsItem) -> PlayerInsight | None:
        """Create PlayerInsight from analysis data."""
        try:
            return PlayerInsight(
                player_name=data['name'],
                insight_type=InsightType(data.get('insight_type', 'general')),
                severity=data.get('severity', 5),
                context=data.get('context', ''),
                confidence=data.get('confidence', 0.5),
                points_adjustment=data.get('points_adjustment', 1.0),
                minutes_adjustment=data.get('minutes_adjustment', 0),
                usage_adjustment=0.0,
                source_news_id=news_item.id,
                timestamp=news_item.timestamp,
                expires_at=time.time() + 86400,  # 24 hours
            )
        except (KeyError, ValueError):
            return None

    def _create_team_insight(self, data: dict, news_item: NewsItem) -> TeamInsight | None:
        """Create TeamInsight from analysis data."""
        try:
            affects = data.get('affects', ['moneyline', 'spread'])
            return TeamInsight(
                team_name=data['name'],
                insight_type=InsightType(data.get('insight_type', 'general')),
                impact_score=data.get('impact_score', 0),
                context=data.get('context', ''),
                confidence=data.get('confidence', 0.5),
                affects_moneyline='moneyline' in affects,
                affects_spread='spread' in affects,
                affects_total='total' in affects,
                source_news_id=news_item.id,
                timestamp=news_item.timestamp,
            )
        except (KeyError, ValueError):
            return None

    def get_player_adjustment(self, player_name: str) -> dict:
        """
        Get adjustment factors for a player.

        Returns:
            Dictionary with adjustment factors
        """
        insight = self.player_impacts.get(player_name)

        if not insight or insight.expires_at < time.time():
            return {
                'has_insight': False,
                'points_multiplier': 1.0,
                'minutes_change': 0,
                'severity': 0,
                'context': '',
            }

        return {
            'has_insight': True,
            'points_multiplier': insight.points_adjustment,
            'minutes_change': insight.minutes_adjustment,
            'severity': insight.severity,
            'context': insight.context,
            'insight_type': insight.insight_type.value,
            'confidence': insight.confidence,
        }

    def get_team_adjustment(self, team_name: str) -> dict:
        """
        Get adjustment factors for a team.

        Returns:
            Dictionary with adjustment factors
        """
        insight = self.team_impacts.get(team_name)

        if not insight:
            return {
                'has_insight': False,
                'impact_score': 0.0,
                'context': '',
            }

        return {
            'has_insight': True,
            'impact_score': insight.impact_score,
            'context': insight.context,
            'insight_type': insight.insight_type.value,
            'affects_moneyline': insight.affects_moneyline,
            'affects_spread': insight.affects_spread,
            'affects_total': insight.affects_total,
            'confidence': insight.confidence,
        }

    def get_injury_boost_factor(self, player_name: str, stat: str = 'points') -> float:
        """
        Get boost factor for player props based on injury context.

        This is designed to integrate with existing injury_fetcher.py logic.

        Args:
            player_name: Player name
            stat: Stat type (points, rebounds, assists, threes)

        Returns:
            Multiplier for projection (< 1 means reduction)
        """
        insight = self.player_impacts.get(player_name)

        if not insight or insight.expires_at < time.time():
            return 1.0

        # Base adjustment from severity
        severity_adj = SEVERITY_ADJUSTMENTS.get(insight.severity, 0)

        # Apply to stat-specific projection
        if insight.insight_type == InsightType.INJURY:
            if stat == 'points':
                return max(0.2, insight.points_adjustment)
            if stat == 'minutes':
                # Convert minutes change to multiplier
                if insight.minutes_adjustment != 0:
                    base_minutes = 28  # Assume ~28 min average
                    return max(0.2, (base_minutes + insight.minutes_adjustment) / base_minutes)
            else:
                # For other stats, use severity-based adjustment
                return max(0.2, 1.0 + severity_adj)

        return 1.0


# =============================================================================
# INTEGRATION PIPELINE
# =============================================================================

class SentimentPipeline:
    """
    Complete pipeline integrating news → analysis → adjustments.

    Usage:
        pipeline = SentimentPipeline()

        # Add news
        pipeline.add_news("LeBron listed questionable with ankle", "...")

        # Process and get adjustments
        pipeline.process()
        adj = pipeline.get_player_adjustment("LeBron James")
    """

    def __init__(self, api_key: str = None):
        self.ingestor = NewsIngestor()
        self.analyzer = SentimentAnalyzer(api_key=api_key)
        self.calculator = InjuryImpactCalculator(self.ingestor, self.analyzer)

    def add_news(
        self,
        headline: str,
        content: str,
        source: str = "manual"
    ) -> NewsItem:
        """Add news item to pipeline."""
        return self.ingestor.add_news(headline, content, source)

    def process(self):
        """Process all unanalyzed news."""
        self.calculator.process_all_news()

    def get_player_adjustment(self, player_name: str) -> dict:
        """Get adjustment for a player."""
        return self.calculator.get_player_adjustment(player_name)

    def get_team_adjustment(self, team_name: str) -> dict:
        """Get adjustment for a team."""
        return self.calculator.get_team_adjustment(team_name)

    def get_injury_boost(self, player_name: str, stat: str = 'points') -> float:
        """Get injury boost factor for player prop."""
        return self.calculator.get_injury_boost_factor(player_name, stat)

    def get_all_active_insights(self) -> dict:
        """Get all active player and team insights."""
        now = time.time()

        return {
            'players': {
                name: {
                    'severity': ins.severity,
                    'context': ins.context,
                    'type': ins.insight_type.value,
                    'points_adj': ins.points_adjustment,
                    'minutes_adj': ins.minutes_adjustment,
                }
                for name, ins in self.calculator.player_impacts.items()
                if ins.expires_at > now
            },
            'teams': {
                name: {
                    'impact': ins.impact_score,
                    'context': ins.context,
                    'type': ins.insight_type.value,
                }
                for name, ins in self.calculator.team_impacts.items()
            },
        }


# =============================================================================
# INJURY FETCHER INTEGRATION
# =============================================================================

def update_injury_adjustments_from_sentiment(
    injury_data: dict,
    pipeline: SentimentPipeline
) -> dict:
    """
    Enhance injury_fetcher.py output with sentiment-based adjustments.

    This function can be called from injury_fetcher.py to incorporate
    qualitative news analysis into injury impact calculations.

    Args:
        injury_data: Dictionary from injury_fetcher.py
        pipeline: SentimentPipeline instance

    Returns:
        Enhanced injury data with sentiment adjustments
    """
    pipeline.process()

    enhanced = injury_data.copy()

    # Enhance player-level injury data
    if 'players' in enhanced:
        for player in enhanced['players']:
            player_name = player.get('name', '')
            adj = pipeline.get_player_adjustment(player_name)

            if adj['has_insight']:
                # Merge sentiment insights with injury data
                player['sentiment_severity'] = adj['severity']
                player['sentiment_context'] = adj['context']
                player['sentiment_points_mult'] = adj['points_multiplier']
                player['sentiment_minutes_adj'] = adj['minutes_change']

                # Potentially upgrade/downgrade injury impact
                current_impact = player.get('impact_score', 0)
                sentiment_impact = -adj['severity'] / 10.0  # Convert to -1 to 0 scale

                # Use worse of the two impacts
                player['combined_impact'] = min(current_impact, sentiment_impact)

    return enhanced


# =============================================================================
# DEMO
# =============================================================================

def demo_sentiment_analysis():
    """Demonstrate news sentiment analysis."""
    print("=" * 70)
    print("NBA NEWS SENTIMENT ANALYSIS")
    print("=" * 70)

    # Initialize pipeline
    pipeline = SentimentPipeline()

    # Add sample news items
    print("\n1. ADDING SAMPLE NEWS")
    print("-" * 40)

    news_items = [
        (
            "LeBron James listed as questionable with ankle soreness",
            "Lakers star LeBron James is listed as questionable for tonight's game "
            "against the Celtics due to ankle soreness. He has been dealing with "
            "the injury for the past week and may be on a minutes restriction "
            "if he plays. Coach Ham said they will make a game-time decision."
        ),
        (
            "Jayson Tatum returning to Indianapolis for first time since college",
            "Celtics forward Jayson Tatum returns to Indiana tonight in what many "
            "are calling a revenge game. Tatum, who played at Duke, has historically "
            "performed well against the Pacers with an average of 32 points in his "
            "last 5 matchups."
        ),
        (
            "Warriors on third game in four nights",
            "The Golden State Warriors face fatigue concerns as they play their "
            "third game in four nights tonight against the Suns. Stephen Curry "
            "played 38 minutes last night and the team flew in late from their "
            "road loss in Denver."
        ),
    ]

    for headline, content in news_items:
        item = pipeline.add_news(headline, content, source="demo")
        print(f"  Added: {headline[:50]}...")
        print(f"    Players: {item.players}")
        print(f"    Teams: {item.teams}")

    # Process news
    print("\n2. PROCESSING NEWS (Fallback Analysis)")
    print("-" * 40)
    pipeline.process()

    # Show insights
    print("\n3. EXTRACTED INSIGHTS")
    print("-" * 40)

    insights = pipeline.get_all_active_insights()

    print("\n  PLAYER INSIGHTS:")
    for name, data in insights['players'].items():
        print(f"    {name}:")
        print(f"      Type: {data['type']}, Severity: {data['severity']}")
        print(f"      Context: {data['context']}")
        print(f"      Points Adj: {data['points_adj']}, Minutes Adj: {data['minutes_adj']}")

    print("\n  TEAM INSIGHTS:")
    for name, data in insights['teams'].items():
        print(f"    {name}:")
        print(f"      Type: {data['type']}, Impact: {data['impact']}")
        print(f"      Context: {data['context']}")

    # Demonstrate adjustment lookups
    print("\n4. ADJUSTMENT LOOKUPS")
    print("-" * 40)

    for player in ["LeBron James", "Jayson Tatum", "Unknown Player"]:
        adj = pipeline.get_player_adjustment(player)
        boost = pipeline.get_injury_boost(player, 'points')
        print(f"\n  {player}:")
        if adj['has_insight']:
            print(f"    Severity: {adj['severity']}")
            print(f"    Points Multiplier: {adj['points_multiplier']:.2f}")
            print(f"    Minutes Change: {adj['minutes_change']}")
            print(f"    Injury Boost Factor: {boost:.2f}")
        else:
            print("    No insights available")

    print("\n5. INTEGRATION EXAMPLE")
    print("-" * 40)
    print("""
    # In daily_predictions.py:
    from news_sentiment import SentimentPipeline

    pipeline = SentimentPipeline()
    pipeline.add_news(headline, content)  # Add latest news
    pipeline.process()

    # Apply to player projections
    for player in players:
        boost = pipeline.get_injury_boost(player['name'], 'points')
        player['projected_points'] *= boost
    """)

    print("\nNews sentiment module ready!")
    return insights


if __name__ == "__main__":
    demo_sentiment_analysis()
