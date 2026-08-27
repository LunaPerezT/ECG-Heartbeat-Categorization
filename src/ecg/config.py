"""Project configuration: paths, Spark tuning and analysis parameters.

Resolution order for every setting, strongest first:

1. explicit keyword arguments passed to :func:`load_config`,
2. environment variables (``ECG_DATA_DIR``, ``ECG_PROCESSED_DIR``,
   ``ECG_REPORTS_DIR``, ``ECG_SPARK_MASTER``, ``ECG_DRIVER_MEMORY``,
   ``ECG_SHUFFLE_PARTITIONS``, ``ECG_SEED``),
3. the YAML file at ``conf/config.yaml``,
4. the dataclass defaults below.

Keeping the raw data *outside* the git working tree is deliberate: the four CSVs
weigh ~583 MB, well past GitHub's file limits, so the default data directory is
the sibling ``data/`` folder of the repository root.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:  # PyYAML is optional: the defaults work without a config file.
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

#: Repository root, resolved from this file's location (``src/ecg/config.py``).
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Default YAML configuration file.
DEFAULT_CONFIG_FILE: Path = REPO_ROOT / "conf" / "config.yaml"

#: File name of each raw CSV, keyed by ``(source, split)``.
RAW_FILES: Dict[str, str] = {
    "mitbih_train": "mitbih_train.csv",
    "mitbih_test": "mitbih_test.csv",
    "ptbdb_normal": "ptbdb_normal.csv",
    "ptbdb_abnormal": "ptbdb_abnormal.csv",
}


@dataclass
class Config:
    """Runtime configuration for the whole pipeline.

    Attributes:
        data_dir: Directory holding the raw CSVs (or a ``raw/`` subfolder with them).
        processed_dir: Destination for Parquet outputs.
        reports_dir: Destination for figures and summary tables.
        spark_master: Spark master URL for local runs; ignored on Databricks.
        driver_memory: ``spark.driver.memory`` for local runs.
        shuffle_partitions: ``spark.sql.shuffle.partitions``; keep it small on a
            laptop, raise it on a cluster.
        arrow_enabled: Enable Arrow for fast ``toPandas`` round trips.
        seed: Global random seed for sampling and splits.
        val_fraction: Fraction of the MIT-BIH training split held out for validation.
        plot_sample_per_class: Number of individual beats drawn per class in the
            example-waveform figures.
    """

    data_dir: Path = field(default_factory=lambda: REPO_ROOT.parent / "data")
    processed_dir: Optional[Path] = None
    reports_dir: Path = field(default_factory=lambda: REPO_ROOT / "reports")

    spark_master: str = "local[*]"
    driver_memory: str = "4g"
    shuffle_partitions: int = 16
    arrow_enabled: bool = True

    seed: int = 42
    val_fraction: float = 0.15
    plot_sample_per_class: int = 6

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.reports_dir = Path(self.reports_dir)
        self.processed_dir = (
            Path(self.processed_dir)
            if self.processed_dir is not None
            else self.data_dir / "processed"
        )

    # ------------------------------------------------------------------ paths

    @property
    def raw_dir(self) -> Path:
        """Directory that actually contains the CSVs.

        Supports both flat layouts (``data/mitbih_train.csv``) and the
        cookiecutter-style ``data/raw/mitbih_train.csv``.
        """
        nested = self.data_dir / "raw"
        if (nested / RAW_FILES["mitbih_train"]).exists():
            return nested
        return self.data_dir

    def raw_path(self, key: str) -> Path:
        """Return the full path of a raw CSV.

        Args:
            key: One of ``mitbih_train``, ``mitbih_test``, ``ptbdb_normal``,
                ``ptbdb_abnormal``.
        """
        try:
            return self.raw_dir / RAW_FILES[key]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"Unknown raw file {key!r}; expected {sorted(RAW_FILES)}") from exc

    @property
    def figures_dir(self) -> Path:
        """Directory for generated figures."""
        return self.reports_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        """Directory for generated summary tables (CSV)."""
        return self.reports_dir / "tables"

    def parquet_path(self, name: str) -> Path:
        """Return the Parquet destination for a named dataset stage."""
        return self.processed_dir / name

    def ensure_dirs(self) -> "Config":
        """Create the output directories if they do not exist yet."""
        for directory in (self.processed_dir, self.figures_dir, self.tables_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------------- validation

    def missing_raw_files(self) -> Dict[str, Path]:
        """Return the raw files that are expected but not present on disk."""
        return {key: self.raw_path(key) for key in RAW_FILES if not self.raw_path(key).exists()}

    def validate_raw(self) -> "Config":
        """Fail loudly when the raw CSVs are not where the config expects them.

        Raises:
            FileNotFoundError: If any of the four CSVs is missing, with a message
                that spells out the expected location and how to override it.
        """
        missing = self.missing_raw_files()
        if missing:
            listed = "\n".join(f"  - {path}" for path in missing.values())
            raise FileNotFoundError(
                "Raw dataset files not found:\n"
                f"{listed}\n\n"
                "Download them from "
                "https://www.kaggle.com/datasets/shayanfazeli/heartbeat and place the "
                f"four CSVs in {self.raw_dir}, or point ECG_DATA_DIR at the folder "
                "that already holds them."
            )
        return self

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON/YAML-friendly view of the configuration."""
        payload = asdict(self)
        return {key: (str(value) if isinstance(value, Path) else value) for key, value in payload.items()}


def _read_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML config file, returning an empty dict when unavailable."""
    if yaml is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):  # pragma: no cover - defensive
        raise ValueError(f"{path} must contain a YAML mapping, got {type(loaded).__name__}")
    return loaded


def _from_env() -> Dict[str, Any]:
    """Collect configuration overrides from environment variables."""
    mapping = {
        "data_dir": os.environ.get("ECG_DATA_DIR"),
        "processed_dir": os.environ.get("ECG_PROCESSED_DIR"),
        "reports_dir": os.environ.get("ECG_REPORTS_DIR"),
        "spark_master": os.environ.get("ECG_SPARK_MASTER"),
        "driver_memory": os.environ.get("ECG_DRIVER_MEMORY"),
        "shuffle_partitions": os.environ.get("ECG_SHUFFLE_PARTITIONS"),
        "seed": os.environ.get("ECG_SEED"),
    }
    overrides = {key: value for key, value in mapping.items() if value not in (None, "")}
    for int_key in ("shuffle_partitions", "seed"):
        if int_key in overrides:
            overrides[int_key] = int(overrides[int_key])
    return overrides


def load_config(config_file: Optional[Path] = None, **overrides: Any) -> Config:
    """Build a :class:`Config`, layering YAML, environment and explicit overrides.

    Args:
        config_file: Path to a YAML file; defaults to ``conf/config.yaml``.
        **overrides: Any :class:`Config` field, taking precedence over everything else.

    Returns:
        A fully resolved configuration object.

    Example:
        >>> cfg = load_config(shuffle_partitions=8)  # doctest: +SKIP
        >>> cfg.raw_path("mitbih_train").name  # doctest: +SKIP
        'mitbih_train.csv'
    """
    path = Path(config_file) if config_file is not None else DEFAULT_CONFIG_FILE
    settings: Dict[str, Any] = {}
    settings.update(_read_yaml(path))
    settings.update(_from_env())
    settings.update({key: value for key, value in overrides.items() if value is not None})

    known = {f for f in Config.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(settings) - known
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")

    return Config(**settings)
