# train_stacking_model.py — Backward-compatible shim
# Real code lives at nba_models/training/train_stacking_model.py
from nba_models.training.train_stacking_model import *  # noqa: F401,F403
from nba_models.training.train_stacking_model import main  # noqa: F811

if __name__ == "__main__":
    main()
