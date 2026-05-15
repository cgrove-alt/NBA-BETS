# railway_retrain.py — Backward-compatible shim
# Real code lives at nba_models/training/railway_retrain.py
#
# IMPORTANT: this shim must explicitly invoke main() under __main__ —
# see odds_tracker_service.py for the failure mode (silent exit on
# script invocation).
from nba_models.training.railway_retrain import *  # noqa: F401,F403
from nba_models.training.railway_retrain import main  # noqa: F811

if __name__ == "__main__":
    main()
