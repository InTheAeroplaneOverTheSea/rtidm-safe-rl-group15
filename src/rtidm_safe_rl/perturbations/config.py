"""Load perturbation pipelines from YAML or JSON configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import yaml

from .base import ObservationPerturbation
from .observation import PersistentCGMBias
from .pipeline import ObservationPerturbationPipeline


PerturbationFactory = Callable[..., ObservationPerturbation]
OBSERVATION_PERTURBATIONS: Dict[str, PerturbationFactory] = {
    PersistentCGMBias.type_name: PersistentCGMBias,
}


def build_perturbation_pipeline(config: Mapping[str, Any]) -> ObservationPerturbationPipeline:
    """Validate a mapping and construct its ordered observation pipeline."""
    version = config.get("version", 1)
    if version != 1:
        raise ValueError(f"Unsupported perturbation config version {version!r}.")

    name = config.get("name", "none")
    if not isinstance(name, str):
        raise TypeError("Perturbation config 'name' must be a string.")

    specifications = config.get("observation", [])
    if not isinstance(specifications, list):
        raise TypeError("Perturbation config 'observation' must be a list.")

    perturbations = []
    for index, specification in enumerate(specifications):
        if not isinstance(specification, Mapping):
            raise TypeError(f"Observation perturbation {index} must be a mapping.")
        type_name = specification.get("type")
        if type_name not in OBSERVATION_PERTURBATIONS:
            available = ", ".join(sorted(OBSERVATION_PERTURBATIONS))
            raise ValueError(
                f"Unknown observation perturbation {type_name!r}; available: {available}."
            )
        parameters = specification.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise TypeError(f"Parameters for {type_name!r} must be a mapping.")
        try:
            perturbations.append(OBSERVATION_PERTURBATIONS[type_name](**dict(parameters)))
        except TypeError as exc:
            raise TypeError(f"Invalid parameters for {type_name!r}: {exc}") from exc

    return ObservationPerturbationPipeline(name=name, perturbations=perturbations)


def load_perturbation_pipeline(path: str | Path) -> ObservationPerturbationPipeline:
    """Load a pipeline from a .yaml, .yml, or .json file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Perturbation config not found: {config_path}")
    with config_path.open(encoding="utf-8") as config_file:
        if config_path.suffix.lower() == ".json":
            config = json.load(config_file)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            config = yaml.safe_load(config_file)
        else:
            raise ValueError("Perturbation config must be YAML or JSON.")
    if not isinstance(config, Mapping):
        raise TypeError("Perturbation config root must be a mapping.")
    return build_perturbation_pipeline(config)
