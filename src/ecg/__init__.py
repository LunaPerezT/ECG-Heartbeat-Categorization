"""ECG Heartbeat Categorization — PySpark data pipeline.

A Spark-native ingestion, exploratory-analysis and preprocessing pipeline for the
`ECG Heartbeat Categorization` dataset (MIT-BIH Arrhythmia + PTB Diagnostic ECG),
designed to run unchanged on a local Spark session and on Databricks.

Typical usage::

    from ecg import get_spark, load_config, ingest_all

    cfg = load_config()
    spark = get_spark()
    frames = ingest_all(spark, cfg)
"""

from ecg.config import Config, load_config
from ecg.session import get_spark, is_databricks, stop_spark

__all__ = [
    "Config",
    "load_config",
    "get_spark",
    "is_databricks",
    "stop_spark",
    "__version__",
]

__version__ = "0.1.0"
