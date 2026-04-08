"""Notebook-facing audit block for AGENTS_NEW."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ENTITY_MAP_ROWS, OUT_DIR, PRIMARY_KEYS
from .io_utils import artifact_exists, ensure_output_dirs, legacy_output_context, read_csv, save_csv
from .plot_utils import plot_bar
from .summary_utils import attach_summary_and_features, display_block_result


def run_audit_block(show_output: bool = True):
    """Run or load the raw audit block and return notebook-friendly artifacts."""

    ensure_output_dirs()
    audit_path = OUT_DIR / "raw_table_audit_AGENT.csv"
    missingness_path = OUT_DIR / "raw_missingness_sample_AGENT.csv"
    key_path = OUT_DIR / "candidate_key_diagnostics_AGENT.csv"
    entity_map_path = OUT_DIR / "entity_map_AGENT.csv"

    from_cache = all(artifact_exists(path) for path in [audit_path, missingness_path, key_path, entity_map_path])
    if from_cache:
        raw_audit = read_csv(audit_path)
        missingness = read_csv(missingness_path)
        key_diagnostics = read_csv(key_path)
        entity_map = read_csv(entity_map_path)
    else:
        with legacy_output_context() as legacy:
            raw_audit, missingness = legacy.build_raw_table_audit()
            key_diagnostics = legacy.build_candidate_key_diagnostics()
        entity_map = pd.DataFrame(ENTITY_MAP_ROWS)
        save_csv(entity_map, entity_map_path)

    plots = [
        plot_bar(
            raw_audit,
            x="table_name",
            y="row_count",
            title="Raw table row counts",
            ylabel="Rows",
            filename="audit_raw_table_rows_AGENTS_NEW.png",
            rotate=70,
        )
    ]
    result = attach_summary_and_features(
        name="Audit block",
        data=raw_audit,
        key_column=PRIMARY_KEYS["audit"],
        from_cache=from_cache,
        validation=key_diagnostics,
        extra_tables={
            "Entity map": entity_map,
            "Top missingness sample": missingness.head(12),
        },
        artifact_paths={
            "audit_table": audit_path,
            "missingness": missingness_path,
            "candidate_keys": key_path,
            "entity_map": entity_map_path,
        },
        plots=plots,
        notes="Audits all raw tables before any entity-level aggregation begins.",
    )
    if show_output:
        display_block_result(result)
    return result
