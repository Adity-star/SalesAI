from pathlib import Path
import yaml

def find_project_root(marker_files=("pyproject.toml", ".git")) -> Path:
    """
    Walk up the directory tree until we find a known project marker like pyproject.toml or .git
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if any((parent / marker).exists() for marker in marker_files):
            return parent
    raise RuntimeError("Project root not found. Make sure you're inside a valid project structure.")

PROJECT_ROOT = find_project_root()

def get_config_path(filename: str = "ml_config.yaml") -> Path:
    """
    Return the absolute path to a config file inside src/config.
    Works both locally and inside containers.
    """
    return PROJECT_ROOT / "src" / "config" / filename

def load_config(filename: str = "ml_config.yaml") -> dict:
    """
    Load a YAML config into a dictionary.
    """
    config_path = get_config_path(filename)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)
