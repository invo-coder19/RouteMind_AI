"""
classifier/predict.py
=====================
Inference module for the LLM Cost Autopilot complexity classifier.

Public API
----------
    classify_and_route(prompt: str) -> tuple[str, ModelConfig]

The function:
1. Loads the trained model from ``classifier/model.pkl`` (once, at import
   time via a module-level singleton).
2. Extracts features from the prompt.
3. Predicts a complexity tier ("simple" / "moderate" / "complex").
4. Looks up the assigned model in ``config/routing_map.yaml``.
5. Resolves the model_id to a ``ModelConfig`` from Phase 1's registry.
6. Returns (tier, model_config).

Design Decisions
----------------
- Model loading happens once at module import time (singleton ``_MODEL``).
  Repeated calls to ``classify_and_route`` do NOT reload the pickle file.
- The routing config is also loaded once at import time (``_ROUTING_MAP``).
  This means config errors (unknown model IDs) surface immediately when the
  module is imported, not on the first prediction call.
- The classifier only knows about tiers — it never sees model names directly.
  The routing config is the only bridge between tiers and model IDs.
"""
from __future__ import annotations

import pathlib
import pickle
from typing import Final

import yaml

# Ensure the project root is importable regardless of CWD.
import sys as _sys
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

from classifier.features import extract_features  # noqa: E402
from models.registry import ModelConfig, get_model  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_CLASSIFIER_DIR: Final[pathlib.Path] = pathlib.Path(__file__).resolve().parent
_MODEL_PKL: Final[pathlib.Path] = _CLASSIFIER_DIR / "model.pkl"
_ROUTING_MAP_YAML: Final[pathlib.Path] = (
    _PROJECT_ROOT / "config" / "routing_map.yaml"
)

# ---------------------------------------------------------------------------
# Valid tiers
# ---------------------------------------------------------------------------
VALID_TIERS: Final[frozenset[str]] = frozenset({"simple", "moderate", "complex"})


# ---------------------------------------------------------------------------
# Module-level singletons (loaded once at import time)
# ---------------------------------------------------------------------------


def _load_model() -> object:
    """Load the trained classifier from disk.  Raises ``FileNotFoundError``
    if ``model.pkl`` has not been generated yet."""
    if not _MODEL_PKL.exists():
        raise FileNotFoundError(
            f"Trained model not found at {_MODEL_PKL}.\n"
            "Run `python data/split_dataset.py` then `python classifier/train.py` first."
        )
    with _MODEL_PKL.open("rb") as fh:
        return pickle.load(fh)


def _load_routing_map() -> dict[str, ModelConfig]:
    """Parse routing_map.yaml and resolve every tier to a ``ModelConfig``.

    Raises
    ------
    FileNotFoundError
        If ``config/routing_map.yaml`` does not exist.
    ValueError
        If the YAML is malformed, a tier is missing, or a model_id is not
        present in the Phase 1 registry.
    """
    if not _ROUTING_MAP_YAML.exists():
        raise FileNotFoundError(
            f"Routing map not found at {_ROUTING_MAP_YAML}.\n"
            "Ensure config/routing_map.yaml exists in the project root."
        )

    with _ROUTING_MAP_YAML.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)

    if not isinstance(raw, dict) or "routing" not in raw:
        raise ValueError(
            f"routing_map.yaml must contain a top-level 'routing' key. Got: {raw!r}"
        )

    routing_section = raw["routing"]
    missing_tiers = VALID_TIERS - set(routing_section.keys())
    if missing_tiers:
        raise ValueError(
            f"routing_map.yaml is missing entries for tiers: {missing_tiers}. "
            f"All of {sorted(VALID_TIERS)} must be defined."
        )

    resolved: dict[str, ModelConfig] = {}
    for tier in VALID_TIERS:
        model_id = routing_section[tier]
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(
                f"routing_map.yaml: tier '{tier}' must map to a non-empty string model_id. "
                f"Got: {model_id!r}"
            )
        try:
            resolved[tier] = get_model(model_id)
        except KeyError as exc:
            raise ValueError(
                f"routing_map.yaml: tier '{tier}' references model_id '{model_id}' "
                f"which does not exist in the Phase 1 model registry.\n"
                f"Original error: {exc}"
            ) from exc

    return resolved


# Load once at module import time — errors surface early.
_MODEL: object = _load_model()
_ROUTING_MAP: dict[str, ModelConfig] = _load_routing_map()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_and_route(prompt: str) -> tuple[str, ModelConfig]:
    """Classify a prompt and return the target model for routing.

    Parameters
    ----------
    prompt:
        The raw prompt string to classify.

    Returns
    -------
    tier : str
        The predicted complexity tier — one of ``"simple"``, ``"moderate"``,
        or ``"complex"``.
    model_config : ModelConfig
        The ``ModelConfig`` from Phase 1's registry assigned to that tier
        by ``config/routing_map.yaml``.

    Raises
    ------
    ValueError
        If the model predicts an unknown tier (should never happen with a
        correctly trained model, but guards against corruption).

    Examples
    --------
    >>> tier, cfg = classify_and_route("What is the capital of France?")
    >>> tier
    'simple'
    >>> cfg.provider
    'ollama'
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string.")

    features = extract_features(prompt)
    feature_df = _features_to_input(features)
    tier: str = str(_MODEL.predict(feature_df)[0])

    if tier not in VALID_TIERS:
        raise ValueError(
            f"Classifier returned unexpected tier '{tier}'. "
            f"Expected one of {sorted(VALID_TIERS)}."
        )

    model_config = _ROUTING_MAP[tier]
    return tier, model_config


def _features_to_input(features: dict) -> "list[dict]":
    """Convert a feature dict to the format expected by sklearn's predict."""
    import pandas as pd
    return pd.DataFrame([features])


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify a prompt and print the routed model."
    )
    parser.add_argument("prompt", nargs="+", help="The prompt text to classify.")
    args = parser.parse_args()
    full_prompt = " ".join(args.prompt)

    tier, cfg = classify_and_route(full_prompt)
    print(f"Prompt   : {full_prompt!r}")
    print(f"Tier     : {tier}")
    print(f"Model    : {cfg.model_id} ({cfg.display_name})")
    print(f"Provider : {cfg.provider}")
    print(f"Cost/1M  : ${cfg.cost_per_input_token * 1_000_000:.4f} in  |  "
          f"${cfg.cost_per_output_token * 1_000_000:.4f} out")
