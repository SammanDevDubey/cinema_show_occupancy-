# src/mp028/findings.py
# ─────────────────────────────────────────────────────────────
# Helpers for reading and updating reports/findings.yaml.
# Each team member edits ONLY their own block.
# ─────────────────────────────────────────────────────────────

from pathlib import Path
import yaml

from mp028.config import load_params, REPO_ROOT


def load_findings(path: Path | None = None) -> dict:
    """Load the current findings.yaml and return its contents.

    Parameters
    ----------
    path : Path, optional
        Override the default findings path.

    Returns
    -------
    dict
        The parsed YAML contents.
    """
    if path is None:
        params = load_params()
        path = REPO_ROOT / params["paths"]["findings"]
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _deep_merge(target: dict, updates: dict) -> dict:
    """Recursively merge *updates* into *target*, preserving existing keys."""
    for key, val in updates.items():
        if isinstance(val, dict) and isinstance(target.get(key), dict):
            _deep_merge(target[key], val)
        else:
            target[key] = val
    return target


def update_findings(block: str, values: dict,
                    path: Path | None = None) -> None:
    """Merge *values* into a (possibly dotted) *block* of findings.yaml.

    Only the keys present in *values* are overwritten; all other keys
    are preserved so that other members' blocks stay safe.

    Parameters
    ----------
    block : str
        Top-level or dotted key, e.g. ``"data_issues"`` or ``"monitoring.psi"``.
    values : dict
        New values to merge into the block.
    path : Path, optional
        Override the default findings path.
    """
    if path is None:
        params = load_params()
        path = REPO_ROOT / params["paths"]["findings"]

    findings = load_findings(path) or {}

    # Walk down the dotted block path (e.g. "monitoring.psi")
    cursor = findings
    for part in block.split("."):
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]

    _deep_merge(cursor, values)

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(findings, fh, default_flow_style=False, sort_keys=False,
                       allow_unicode=True)
