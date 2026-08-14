"""
Loads YAML configs into plain dicts (kept simple deliberately — no need for
a heavier framework like Hydra at this project's scale). Every script imports
from here instead of hardcoding paths or hyperparameters.
"""

from pathlib import Path
import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def load_config(name: str) -> dict:
    """
    Load a config file by name, e.g. load_config("training") -> configs/training.yaml

    Raises FileNotFoundError with a clear message if the config doesn't exist,
    rather than a cryptic yaml error.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"No config found at {path}. Available configs: "
            f"{[p.stem for p in CONFIG_DIR.glob('*.yaml')]}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def load_all() -> dict:
    """Convenience: load model + training + eval configs together."""
    return {
        "model": load_config("model"),
        "training": load_config("training"),
        "eval": load_config("eval"),
    }
