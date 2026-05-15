# odds_tracker_service.py — Backward-compatible shim
# Real code lives at nba_betting/odds/odds_tracker_service.py
#
# IMPORTANT: this shim must explicitly invoke main() under __main__.
# Without the bottom block, `python odds_tracker_service.py` imports the
# module then exits without doing anything — Railway's "Starting Container"
# then deploymentStopped=True. (Fixed 2026-05-15 after Odds Tracker was
# silently dead in production.)
from nba_betting.odds.odds_tracker_service import *  # noqa: F401,F403
from nba_betting.odds.odds_tracker_service import main  # noqa: F811

if __name__ == "__main__":
    main()
