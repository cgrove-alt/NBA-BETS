"""
Model Registry for Continuous Learning

Manages model versions with:
1. Version tracking with metadata and metrics
2. Model comparison between versions
3. Rollback capability to previous versions
4. Active model selection
"""

import json
import shutil
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelVersion:
    """Represents a model version in the registry."""
    version_id: str
    model_type: str
    model_path: str
    metrics: Dict
    training_date: str
    is_active: bool
    feature_names: List[str] = None
    hyperparameters: Dict = None
    training_samples: int = 0
    notes: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


class ModelRegistry:
    """Manages model versions for tracking and rollback."""

    def __init__(self, base_path: str = None):
        """Initialize model registry.

        Args:
            base_path: Base path for storing models. Defaults to 'models' directory.
        """
        if base_path is None:
            base_path = str(Path(__file__).parent.parent / "models")

        self.base_path = Path(base_path)
        self.registry_file = self.base_path / "registry.json"
        self._ensure_dirs()

    def _ensure_dirs(self):
        """Ensure required directories exist."""
        self.base_path.mkdir(exist_ok=True)
        (self.base_path / "versions").mkdir(exist_ok=True)

    def _load_registry(self) -> Dict:
        """Load registry from JSON file."""
        if self.registry_file.exists():
            try:
                return json.loads(self.registry_file.read_text())
            except json.JSONDecodeError:
                print(f"Warning: Registry file corrupted, starting fresh")
                return {}
        return {}

    def _save_registry(self, registry: Dict):
        """Save registry to JSON file."""
        self.registry_file.write_text(json.dumps(registry, indent=2, default=str))

    def register_model(
        self,
        model_type: str,
        model_path: str,
        metrics: Dict,
        training_date: str = None,
        feature_names: List[str] = None,
        hyperparameters: Dict = None,
        training_samples: int = 0,
        notes: str = "",
        make_active: bool = True,
    ) -> str:
        """Register a new model version.

        Args:
            model_type: Type of model (e.g., 'moneyline', 'spread', 'player_points')
            model_path: Path to the model file
            metrics: Training/validation metrics
            training_date: Date model was trained (defaults to now)
            feature_names: List of feature names used
            hyperparameters: Model hyperparameters
            training_samples: Number of training samples used
            notes: Optional notes about this version
            make_active: Whether to make this the active model

        Returns:
            version_id: Unique identifier for this version
        """
        version_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create versioned directory
        version_dir = self.base_path / "versions" / model_type / version_id
        version_dir.mkdir(parents=True, exist_ok=True)

        # Copy model to versioned location
        src_path = Path(model_path)
        if src_path.exists():
            dest_path = version_dir / src_path.name
            shutil.copy(src_path, dest_path)
            stored_path = str(dest_path)
        else:
            stored_path = model_path

        # Create version record
        version = ModelVersion(
            version_id=version_id,
            model_type=model_type,
            model_path=stored_path,
            metrics=metrics,
            training_date=training_date or datetime.now().isoformat(),
            is_active=make_active,
            feature_names=feature_names or [],
            hyperparameters=hyperparameters or {},
            training_samples=training_samples,
            notes=notes,
        )

        # Update registry
        registry = self._load_registry()

        if model_type not in registry:
            registry[model_type] = []

        # Deactivate previous versions if making this active
        if make_active:
            for v in registry[model_type]:
                v['is_active'] = False

        registry[model_type].append(version.to_dict())
        self._save_registry(registry)

        print(f"Registered model {model_type} version {version_id}")
        return version_id

    def get_active_model(self, model_type: str) -> Optional[Dict]:
        """Get currently active model for a type.

        Args:
            model_type: Type of model to retrieve

        Returns:
            Active model version dict or None
        """
        registry = self._load_registry()
        versions = registry.get(model_type, [])

        for v in reversed(versions):
            if v.get('is_active'):
                return v

        # If no active model, return the latest
        if versions:
            return versions[-1]

        return None

    def get_all_versions(self, model_type: str) -> List[Dict]:
        """Get all versions for a model type.

        Args:
            model_type: Type of model

        Returns:
            List of all version records
        """
        registry = self._load_registry()
        return registry.get(model_type, [])

    def get_version(self, model_type: str, version_id: str) -> Optional[Dict]:
        """Get a specific model version.

        Args:
            model_type: Type of model
            version_id: Version identifier

        Returns:
            Version dict or None
        """
        versions = self.get_all_versions(model_type)
        for v in versions:
            if v.get('version_id') == version_id:
                return v
        return None

    def activate_version(self, model_type: str, version_id: str) -> bool:
        """Activate a specific model version.

        Args:
            model_type: Type of model
            version_id: Version to activate

        Returns:
            True if successful
        """
        registry = self._load_registry()
        versions = registry.get(model_type, [])

        found = False
        for v in versions:
            if v.get('version_id') == version_id:
                v['is_active'] = True
                found = True
            else:
                v['is_active'] = False

        if found:
            self._save_registry(registry)
            print(f"Activated {model_type} version {version_id}")

        return found

    def rollback(self, model_type: str, version_id: str) -> bool:
        """Rollback to a previous model version.

        This activates the specified version and optionally copies it
        to the main model location.

        Args:
            model_type: Type of model
            version_id: Version to rollback to

        Returns:
            True if successful
        """
        version = self.get_version(model_type, version_id)
        if not version:
            print(f"Version {version_id} not found for {model_type}")
            return False

        # Activate the version
        self.activate_version(model_type, version_id)

        # Copy to main model location
        model_map = {
            'moneyline': 'moneyline_ensemble.pkl',
            'spread': 'spread_ensemble.pkl',
            'player_points': 'player_points.pkl',
            'player_rebounds': 'player_rebounds.pkl',
            'player_assists': 'player_assists.pkl',
            'player_threes': 'player_threes.pkl',
            'player_pra': 'player_pra.pkl',
        }

        if model_type in model_map:
            src = Path(version['model_path'])
            dest = self.base_path / model_map[model_type]

            if src.exists():
                shutil.copy(src, dest)
                print(f"Rolled back {model_type} to version {version_id}")
                return True

        return True

    def compare_versions(
        self,
        model_type: str,
        version_a: str,
        version_b: str
    ) -> Dict:
        """Compare metrics between two versions.

        Args:
            model_type: Type of model
            version_a: First version to compare
            version_b: Second version to compare

        Returns:
            Comparison dict with metrics from both versions
        """
        registry = self._load_registry()
        versions = {v['version_id']: v for v in registry.get(model_type, [])}

        a_data = versions.get(version_a, {})
        b_data = versions.get(version_b, {})

        a_metrics = a_data.get('metrics', {})
        b_metrics = b_data.get('metrics', {})

        # Calculate differences
        comparison = {}
        all_keys = set(a_metrics.keys()) | set(b_metrics.keys())

        for key in all_keys:
            a_val = a_metrics.get(key)
            b_val = b_metrics.get(key)

            if isinstance(a_val, (int, float)) and isinstance(b_val, (int, float)):
                diff = b_val - a_val
                pct_change = ((b_val - a_val) / a_val * 100) if a_val != 0 else 0
                comparison[key] = {
                    'version_a': a_val,
                    'version_b': b_val,
                    'difference': round(diff, 4),
                    'pct_change': round(pct_change, 2),
                }

        return {
            'model_type': model_type,
            'version_a': version_a,
            'version_b': version_b,
            'version_a_date': a_data.get('training_date'),
            'version_b_date': b_data.get('training_date'),
            'metrics_comparison': comparison,
            'version_a_samples': a_data.get('training_samples', 0),
            'version_b_samples': b_data.get('training_samples', 0),
        }

    def load_model(self, model_type: str, version_id: str = None) -> Optional[Any]:
        """Load a model from the registry.

        Args:
            model_type: Type of model
            version_id: Specific version to load (defaults to active)

        Returns:
            Loaded model object or None
        """
        if version_id:
            version = self.get_version(model_type, version_id)
        else:
            version = self.get_active_model(model_type)

        if not version:
            return None

        model_path = Path(version['model_path'])
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            return None

        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def get_model_types(self) -> List[str]:
        """Get all model types in the registry.

        Returns:
            List of model type names
        """
        registry = self._load_registry()
        return list(registry.keys())

    def cleanup_old_versions(self, model_type: str, keep_count: int = 5):
        """Remove old model versions, keeping the N most recent.

        Args:
            model_type: Type of model to clean up
            keep_count: Number of versions to keep
        """
        registry = self._load_registry()
        versions = registry.get(model_type, [])

        if len(versions) <= keep_count:
            return

        # Sort by version_id (timestamp-based) and keep most recent
        sorted_versions = sorted(versions, key=lambda v: v['version_id'], reverse=True)
        versions_to_remove = sorted_versions[keep_count:]

        # Don't remove active version
        versions_to_remove = [v for v in versions_to_remove if not v.get('is_active')]

        for v in versions_to_remove:
            # Remove model files
            model_path = Path(v['model_path'])
            if model_path.exists():
                try:
                    model_path.unlink()
                    # Try to remove empty parent directory
                    model_path.parent.rmdir()
                except Exception as e:
                    print(f"Error removing model file: {e}")

        # Update registry
        kept_version_ids = {v['version_id'] for v in sorted_versions[:keep_count]}
        registry[model_type] = [v for v in versions if v['version_id'] in kept_version_ids]
        self._save_registry(registry)

        print(f"Cleaned up {len(versions_to_remove)} old versions of {model_type}")

    def print_status(self):
        """Print a formatted status of the registry."""
        registry = self._load_registry()

        print(f"\n{'='*60}")
        print(f"MODEL REGISTRY STATUS")
        print(f"{'='*60}")
        print(f"Location: {self.base_path}")
        print(f"Model Types: {len(registry)}")

        for model_type, versions in registry.items():
            active = next((v for v in versions if v.get('is_active')), None)
            print(f"\n{model_type}:")
            print(f"  Total versions: {len(versions)}")
            if active:
                print(f"  Active: {active['version_id']}")
                print(f"  Training date: {active['training_date']}")
                metrics = active.get('metrics', {})
                if metrics:
                    key_metrics = ['accuracy', 'rmse', 'r2', 'win_rate']
                    for key in key_metrics:
                        if key in metrics:
                            print(f"  {key}: {metrics[key]:.4f}")

        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # CLI for model registry operations
    import argparse

    parser = argparse.ArgumentParser(description="Model Registry CLI")
    parser.add_argument("--status", action="store_true", help="Show registry status")
    parser.add_argument("--list", type=str, help="List versions for a model type")
    parser.add_argument("--compare", nargs=3, metavar=('TYPE', 'V1', 'V2'),
                        help="Compare two versions")
    parser.add_argument("--rollback", nargs=2, metavar=('TYPE', 'VERSION'),
                        help="Rollback to a specific version")
    parser.add_argument("--cleanup", nargs=2, metavar=('TYPE', 'KEEP_COUNT'),
                        help="Clean up old versions")

    args = parser.parse_args()

    registry = ModelRegistry()

    if args.status:
        registry.print_status()
    elif args.list:
        versions = registry.get_all_versions(args.list)
        for v in versions:
            active = "✓" if v.get('is_active') else " "
            print(f"  [{active}] {v['version_id']} - {v['training_date']}")
    elif args.compare:
        model_type, v1, v2 = args.compare
        result = registry.compare_versions(model_type, v1, v2)
        print(json.dumps(result, indent=2))
    elif args.rollback:
        model_type, version = args.rollback
        registry.rollback(model_type, version)
    elif args.cleanup:
        model_type, keep_count = args.cleanup
        registry.cleanup_old_versions(model_type, int(keep_count))
    else:
        parser.print_help()
