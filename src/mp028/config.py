# src/mp028/config.py
# ─────────────────────────────────────────────────────────────
# Loads params.yaml from the repository root so every module
# reads the same settings without hard-coding paths or values.
# ─────────────────────────────────────────────────────────────

from pathlib import Path
import yaml


def _find_repo_root() -> Path:
    """Walk up from this file until we find params.yaml."""
    current = Path(__file__).resolve().parent          # src/mp028/
    for ancestor in [current] + list(current.parents):
        if (ancestor / "params.yaml").exists():
            return ancestor
    # Fallback: current working directory
    return Path.cwd()


# Repo root is computed once at import time.
REPO_ROOT = _find_repo_root()


def load_params(path: Path | None = None) -> dict:
    """Read params.yaml and return the full dict.

    Parameters
    ----------
    path : Path, optional
        Override the default params.yaml location (useful in tests).

    Returns
    -------
    dict
        The parsed YAML contents.
    """
    if path is None:
        path = REPO_ROOT / "params.yaml"
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
