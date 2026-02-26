# scheduled_retraining.py — Backward-compatible shim
# Real code lives at nba_models/training/scheduled_retraining.py
from nba_models.training.scheduled_retraining import *  # noqa: F401,F403
from nba_models.training.scheduled_retraining import main  # noqa: F811

if __name__ == "__main__":
    main()
