# daily_predictions.py — Backward-compatible shim
# Real code lives at nba_models/inference/daily_predictions.py
from nba_models.inference.daily_predictions import *  # noqa: F401,F403

if __name__ == "__main__":
    main()  # noqa: F405
