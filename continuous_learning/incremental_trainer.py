"""
Incremental Trainer for Continuous Learning

Supports incremental model updates using:
1. XGBoost warm start (continue training from existing model)
2. LightGBM continue_train
3. Time-decay weighting for recent data emphasis
4. Feature extraction from settled predictions
"""

import sys
import pickle
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add parent for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from prop_tracker import PropTracker
from .model_registry import ModelRegistry

# Try to import XGBoost and LightGBM
try:
    import xgboost as xgb
    from xgboost import XGBRegressor, XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not available for incremental training")

try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor, LGBMClassifier
    HAS_LIGHTGBM = True
except (ImportError, OSError):
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not available for incremental training")


class IncrementalTrainer:
    """Handles incremental model training with new prediction data."""

    # Time decay configuration
    TIME_DECAY_HALF_LIFE_DAYS = 180  # Recent data weighted 2x every 180 days

    def __init__(
        self,
        prop_tracker: PropTracker = None,
        model_registry: ModelRegistry = None,
        model_dir: str = None
    ):
        """Initialize incremental trainer.

        Args:
            prop_tracker: PropTracker for accessing settled predictions
            model_registry: ModelRegistry for version control
            model_dir: Directory containing models
        """
        self.prop_tracker = prop_tracker or PropTracker()
        self.registry = model_registry or ModelRegistry()

        if model_dir is None:
            model_dir = str(Path(__file__).parent.parent / "models")
        self.model_dir = Path(model_dir)

    def get_training_data_from_predictions(
        self,
        prop_type: str = None,
        min_samples: int = 100,
        max_days: int = 365
    ) -> Optional[pd.DataFrame]:
        """Extract training data from settled predictions.

        Args:
            prop_type: Filter by prop type (points, rebounds, etc.)
            min_samples: Minimum samples required
            max_days: Maximum days to look back

        Returns:
            DataFrame with training features and targets, or None if insufficient data
        """
        cutoff_date = (datetime.now() - timedelta(days=max_days)).strftime("%Y-%m-%d")

        conn = sqlite3.connect(self.prop_tracker.db_path)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT * FROM prop_predictions
            WHERE is_settled = 1 AND hit >= 0
              AND game_date >= ?
        """
        params = [cutoff_date]

        if prop_type:
            query += " AND prop_type = ?"
            params.append(prop_type.lower())

        query += " ORDER BY game_date DESC"

        cursor = conn.execute(query, params)
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        if len(rows) < min_samples:
            print(f"Insufficient data: {len(rows)} samples (need {min_samples})")
            return None

        # Convert to DataFrame
        df = pd.DataFrame(rows)

        # Calculate sample weights with time decay
        df['sample_weight'] = df['game_date'].apply(self._calculate_time_weight)

        return df

    def _calculate_time_weight(self, date_str: str) -> float:
        """Calculate time-decay weight for a sample.

        More recent samples get higher weight.

        Args:
            date_str: Date string (YYYY-MM-DD)

        Returns:
            Sample weight (1.0 = current, decreases with age)
        """
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            days_ago = (datetime.now() - date).days

            # Exponential decay with configurable half-life
            weight = 0.5 ** (days_ago / self.TIME_DECAY_HALF_LIFE_DAYS)
            return max(0.1, weight)  # Minimum weight of 0.1
        except Exception:
            return 0.5  # Default weight

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for training.

        Args:
            df: DataFrame with prediction data
            feature_cols: Columns to use as features

        Returns:
            Tuple of (X features, y targets, sample_weights)
        """
        if feature_cols is None:
            # Default feature columns from prediction data
            feature_cols = [
                'predicted_value',
                'market_line',
                'edge_pct',
                'confidence',
                'opp_def_rating',
                'opp_adjustment',
            ]

        # Use available columns
        available_cols = [c for c in feature_cols if c in df.columns]

        X = df[available_cols].fillna(0).values
        y = df['actual_value'].values
        weights = df['sample_weight'].values

        return X, y, weights

    def incremental_train_xgboost(
        self,
        model_type: str,
        new_data: pd.DataFrame,
        existing_model_path: str = None,
        n_estimators_boost: int = 50,
    ) -> Tuple[Any, Dict]:
        """Incrementally train an XGBoost model.

        Args:
            model_type: Type of model (e.g., 'player_points')
            new_data: DataFrame with new training data
            existing_model_path: Path to existing model (optional)
            n_estimators_boost: Additional trees to add

        Returns:
            Tuple of (trained model, metrics dict)
        """
        if not HAS_XGBOOST:
            raise ImportError("XGBoost is required for incremental training")

        # Prepare features
        X, y, weights = self.prepare_features(new_data)

        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )

        # Load existing model or create new one
        if existing_model_path and Path(existing_model_path).exists():
            print(f"Loading existing model from {existing_model_path}")
            model = xgb.XGBRegressor()
            model.load_model(existing_model_path)

            # Continue training with new data
            model.set_params(n_estimators=model.n_estimators + n_estimators_boost)
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                xgb_model=model.get_booster(),
                verbose=False
            )
        else:
            print("Training new XGBoost model")
            model = xgb.XGBRegressor(
                n_estimators=100 + n_estimators_boost,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
            model.fit(
                X_train, y_train,
                sample_weight=w_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

        # Calculate metrics
        y_pred = model.predict(X_val)
        metrics = {
            'rmse': float(np.sqrt(np.mean((y_val - y_pred) ** 2))),
            'mae': float(np.mean(np.abs(y_val - y_pred))),
            'r2': float(1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
        }

        return model, metrics

    def incremental_train_lightgbm(
        self,
        model_type: str,
        new_data: pd.DataFrame,
        existing_model_path: str = None,
        num_boost_round: int = 50,
    ) -> Tuple[Any, Dict]:
        """Incrementally train a LightGBM model.

        Args:
            model_type: Type of model (e.g., 'player_points')
            new_data: DataFrame with new training data
            existing_model_path: Path to existing model (optional)
            num_boost_round: Additional boosting rounds

        Returns:
            Tuple of (trained model, metrics dict)
        """
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM is required for incremental training")

        # Prepare features
        X, y, weights = self.prepare_features(new_data)

        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train, weight=w_train)
        val_data = lgb.Dataset(X_val, label=y_val, weight=w_val, reference=train_data)

        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }

        # Load existing model or train fresh
        if existing_model_path and Path(existing_model_path).exists():
            print(f"Continuing training from {existing_model_path}")
            model = lgb.train(
                params,
                train_data,
                num_boost_round=num_boost_round,
                valid_sets=[val_data],
                init_model=existing_model_path,
            )
        else:
            print("Training new LightGBM model")
            model = lgb.train(
                params,
                train_data,
                num_boost_round=100 + num_boost_round,
                valid_sets=[val_data],
            )

        # Calculate metrics
        y_pred = model.predict(X_val)
        metrics = {
            'rmse': float(np.sqrt(np.mean((y_val - y_pred) ** 2))),
            'mae': float(np.mean(np.abs(y_val - y_pred))),
            'r2': float(1 - np.sum((y_val - y_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)),
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'num_trees': model.num_trees(),
        }

        return model, metrics

    def retrain_prop_model(
        self,
        prop_type: str,
        min_samples: int = 100,
        use_library: str = 'xgboost'
    ) -> Optional[str]:
        """Retrain a player prop model with new data.

        Args:
            prop_type: Type of prop (points, rebounds, assists, threes, pra)
            min_samples: Minimum samples required
            use_library: Which library to use ('xgboost' or 'lightgbm')

        Returns:
            Version ID if successful, None otherwise
        """
        # Get training data from settled predictions
        training_data = self.get_training_data_from_predictions(
            prop_type=prop_type,
            min_samples=min_samples
        )

        if training_data is None:
            print(f"Insufficient training data for {prop_type}")
            return None

        print(f"Training {prop_type} model with {len(training_data)} samples")

        # Get existing model path
        model_name = f"player_{prop_type}.pkl"
        existing_path = self.model_dir / model_name

        # Train with selected library
        if use_library == 'xgboost' and HAS_XGBOOST:
            model, metrics = self.incremental_train_xgboost(
                model_type=prop_type,
                new_data=training_data,
                existing_model_path=str(existing_path) if existing_path.exists() else None,
            )
        elif use_library == 'lightgbm' and HAS_LIGHTGBM:
            model, metrics = self.incremental_train_lightgbm(
                model_type=prop_type,
                new_data=training_data,
                existing_model_path=str(existing_path) if existing_path.exists() else None,
            )
        else:
            print(f"Library {use_library} not available")
            return None

        # Save model
        new_model_path = self.model_dir / model_name
        if use_library == 'xgboost':
            model.save_model(str(new_model_path))
        else:
            model.save_model(str(new_model_path))

        # Register in model registry
        version_id = self.registry.register_model(
            model_type=f"player_{prop_type}",
            model_path=str(new_model_path),
            metrics=metrics,
            training_samples=len(training_data),
            notes=f"Incremental training with {use_library}",
        )

        print(f"Model {prop_type} saved as version {version_id}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")

        return version_id

    def retrain_all_prop_models(self, min_samples: int = 50) -> Dict[str, str]:
        """Retrain all player prop models.

        Args:
            min_samples: Minimum samples per prop type

        Returns:
            Dict mapping prop_type to version_id
        """
        prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']
        results = {}

        for prop_type in prop_types:
            version_id = self.retrain_prop_model(prop_type, min_samples)
            if version_id:
                results[prop_type] = version_id

        return results

    def should_retrain(self, prop_type: str, min_new_samples: int = 50) -> bool:
        """Check if a model should be retrained based on new data.

        Args:
            prop_type: Type of prop model
            min_new_samples: Minimum new samples to warrant retraining

        Returns:
            True if retraining is recommended
        """
        # Get last training date from registry
        active_model = self.registry.get_active_model(f"player_{prop_type}")

        if not active_model:
            return True  # No model exists

        last_training = active_model.get('training_date', '')
        last_samples = active_model.get('training_samples', 0)

        # Get new settled predictions since last training
        new_data = self.get_training_data_from_predictions(prop_type=prop_type)

        if new_data is None:
            return False

        new_count = len(new_data)

        # Retrain if we have significantly more data
        if new_count >= last_samples + min_new_samples:
            return True

        return False

    def get_training_status(self) -> Dict:
        """Get status of training data availability.

        Returns:
            Dict with sample counts per prop type
        """
        prop_types = ['points', 'rebounds', 'assists', 'threes', 'pra']
        status = {}

        for prop_type in prop_types:
            data = self.get_training_data_from_predictions(
                prop_type=prop_type,
                min_samples=1  # Get any available data
            )
            status[prop_type] = {
                'samples': len(data) if data is not None else 0,
                'should_retrain': self.should_retrain(prop_type),
            }

        return status


def run_incremental_training():
    """Convenience function to run incremental training."""
    trainer = IncrementalTrainer()
    status = trainer.get_training_status()

    print("\nTraining Data Status:")
    for prop_type, info in status.items():
        print(f"  {prop_type}: {info['samples']} samples, retrain: {info['should_retrain']}")

    results = trainer.retrain_all_prop_models()
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Incremental Model Training")
    parser.add_argument("--status", action="store_true", help="Show training data status")
    parser.add_argument("--train", type=str, help="Train specific prop type")
    parser.add_argument("--train-all", action="store_true", help="Train all models")
    parser.add_argument("--min-samples", type=int, default=50, help="Minimum samples")
    parser.add_argument("--library", choices=['xgboost', 'lightgbm'], default='xgboost',
                        help="ML library to use")

    args = parser.parse_args()

    trainer = IncrementalTrainer()

    if args.status:
        status = trainer.get_training_status()
        print("\nTraining Data Status:")
        for prop_type, info in status.items():
            print(f"  {prop_type}: {info['samples']} samples, retrain: {info['should_retrain']}")
    elif args.train:
        trainer.retrain_prop_model(args.train, args.min_samples, args.library)
    elif args.train_all:
        results = trainer.retrain_all_prop_models(args.min_samples)
        print(f"\nRetrained {len(results)} models")
    else:
        parser.print_help()
