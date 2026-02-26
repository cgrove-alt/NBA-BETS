# train_complete_balldontlie.py — Backward-compatible shim
# Real code lives at nba_models/training/train_complete_balldontlie.py
from nba_models.training.train_complete_balldontlie import *  # noqa: F401,F403
from nba_models.training.train_complete_balldontlie import main  # noqa: F811

if __name__ == "__main__":
    main()
