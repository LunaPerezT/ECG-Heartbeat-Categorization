"""SparkSession management for both local runs and Databricks.

On Databricks a session already exists and its configuration is owned by the
cluster, so :func:`get_spark` attaches to it instead of building a new one. Only
outside Databricks does it set a master, driver memory and shuffle parallelism.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from pyspark.sql import SparkSession

from ecg.config import Config

#: Application name used for local sessions.
APP_NAME = "ecg-heartbeat-categorization"


def is_databricks() -> bool:
    """Return ``True`` when the code is running on a Databricks cluster.

    Databricks sets ``DATABRICKS_RUNTIME_VERSION`` in every driver and executor
    environment, which makes it the cheapest reliable probe.
    """
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


def get_spark(
    cfg: Optional[Config] = None,
    app_name: str = APP_NAME,
    extra_conf: Optional[Dict[str, Any]] = None,
) -> SparkSession:
    """Return a ready-to-use :class:`~pyspark.sql.SparkSession`.

    Args:
        cfg: Project configuration; only its Spark fields are read. Defaults are
            used when omitted.
        app_name: Application name for local sessions.
        extra_conf: Extra ``spark.*`` options applied on top of the defaults.

    Returns:
        An active session. On Databricks this is the cluster's session, so the
        caller must not stop it.

    Note:
        ``spark.sql.execution.arrow.pyspark.enabled`` is switched on because every
        EDA helper in :mod:`ecg.eda` finishes with a small ``toPandas()`` call;
        Arrow makes that transfer roughly an order of magnitude cheaper.
    """
    cfg = cfg or Config()

    if is_databricks():
        session = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    else:
        builder = (
            SparkSession.builder.appName(app_name)
            .master(cfg.spark_master)
            .config("spark.driver.memory", cfg.driver_memory)
            .config("spark.sql.shuffle.partitions", cfg.shuffle_partitions)
            .config("spark.driver.maxResultSize", "1g")
            # Reading 583 MB of wide CSV benefits from bigger input splits.
            .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))
            .config("spark.ui.showConsoleProgress", "false")
        )
        session = builder.getOrCreate()

    session.conf.set("spark.sql.execution.arrow.pyspark.enabled", str(cfg.arrow_enabled).lower())
    session.conf.set("spark.sql.execution.arrow.pyspark.fallback.enabled", "true")

    for key, value in (extra_conf or {}).items():
        session.conf.set(key, value)

    return session


def stop_spark(session: Optional[SparkSession] = None) -> None:
    """Stop a local session; a no-op on Databricks.

    Args:
        session: Session to stop. The active session is used when omitted.
    """
    if is_databricks():
        return
    session = session or SparkSession.getActiveSession()
    if session is not None:
        session.stop()


def session_summary(session: SparkSession) -> Dict[str, Any]:
    """Return a small dictionary describing the running session.

    Useful as the first cell of a notebook so the executed output records which
    environment produced the results.
    """
    context = session.sparkContext
    return {
        "spark_version": session.version,
        "application_id": context.applicationId,
        "master": context.master,
        "default_parallelism": context.defaultParallelism,
        "shuffle_partitions": session.conf.get("spark.sql.shuffle.partitions"),
        "arrow_enabled": session.conf.get("spark.sql.execution.arrow.pyspark.enabled"),
        "databricks": is_databricks(),
    }
