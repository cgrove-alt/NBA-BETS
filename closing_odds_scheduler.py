# closing_odds_scheduler.py — Backward-compatible shim
# Real code lives at nba_betting/odds/closing_odds_scheduler.py
#
# IMPORTANT: this shim must explicitly invoke main() under __main__ —
# see odds_tracker_service.py for the failure mode (silent exit on
# script invocation). The closing-odds scheduler is what populates the
# closing_odds column for CLV measurement; without main() invocation
# any cron schedule pointing at this file silently does nothing.
from nba_betting.odds.closing_odds_scheduler import *  # noqa: F401,F403
from nba_betting.odds.closing_odds_scheduler import main  # noqa: F811

if __name__ == "__main__":
    main()
