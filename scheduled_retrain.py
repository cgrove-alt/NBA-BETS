# scheduled_retrain.py — Backward-compatible shim
# Real code lives at nba_models/training/scheduled_retrain.py
from nba_models.training.scheduled_retrain import *  # noqa: F401,F403
from nba_models.training.scheduled_retrain import main  # noqa: F811

if __name__ == "__main__":
    main()
