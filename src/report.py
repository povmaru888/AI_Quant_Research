"""P4-08: research report exporter (SDD 14.2 報表).

Exports performance.csv, factor_ic.csv, run_summary.json, and a
single-page report.pdf for one completed run. The PDF is hand-written
minimal PDF 1.4 (no new dependencies); rich chart PDFs stay a Phase 5
option once a static-export backend is pinned. Only ``succeeded`` runs
export; anything else raises ``ValueError``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

SUCCEEDED = "succeeded"

SUMMARY_KEYS: tuple[str, ...] = (
    "run_id",
    "data_end_date",
    "feature_version",
    "model_version",
    "parameter_version",
)


@dataclass(frozen=True)
class ReportPaths:
    """Locations of the four exported report files."""

    performance_csv: Path
    factor_ic_csv: Path
    run_summary_json: Path
    report_pdf: Path


class ReportStore(Protocol):
    """Persistence seam for report exports (Phase 5 implements with DB)."""

    def get_run_status(self, run_id: str) -> str: ...
    def load_run_summary(self, run_id: str) -> dict: ...
    def load_performance(self, run_id: str) -> pd.DataFrame: ...
    def load_factor_ic(self, run_id: str) -> pd.DataFrame: ...


def export_reports(run_id: str, output_dir: str | Path, store: ReportStore) -> ReportPaths:
    """Export the four report files for one completed run."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError(f"invalid run_id: must be a non-empty string, got {run_id!r}")
    try:
        status = store.get_run_status(run_id)
    except Exception as exc:
        raise ValueError(f"invalid run_id: unknown run {run_id!r}") from exc
    if status != SUCCEEDED:
        raise ValueError(f"invalid run_id: run {run_id!r} is not completed ({status!r})")
    summary = store.load_run_summary(run_id)
    if not isinstance(summary, dict):
        raise ValueError("invalid run summary: must be a dict")
    missing = [k for k in SUMMARY_KEYS if k not in summary]
    if missing:
        raise ValueError(f"invalid run summary: missing {missing}")
    performance = _require_columns(store.load_performance(run_id), ("date", "nav"), "performance")
    factor_ic = _require_columns(store.load_factor_ic(run_id), ("factor", "ic"), "factor_ic")

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths = ReportPaths(
        performance_csv=target / "performance.csv",
        factor_ic_csv=target / "factor_ic.csv",
        run_summary_json=target / "run_summary.json",
        report_pdf=target / "report.pdf",
    )
    performance.to_csv(paths.performance_csv, index=False)
    factor_ic.to_csv(paths.factor_ic_csv, index=False)
    paths.run_summary_json.write_text(
        _summary_json(summary, performance, factor_ic), encoding="utf-8"
    )
    paths.report_pdf.write_bytes(_report_pdf(summary))
    return paths


def _require_columns(frame: object, columns: tuple[str, ...], name: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"invalid {name}: must be a DataFrame")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"invalid {name}: missing columns {missing}")
    return frame.reset_index(drop=True)


def _summary_json(summary: dict, performance: pd.DataFrame, factor_ic: pd.DataFrame) -> str:
    payload = {key: summary[key] for key in SUMMARY_KEYS}
    payload["metrics"] = summary.get("metrics", {})
    payload["oos_months"] = summary.get("oos_months", [])
    payload["performance_rows"] = int(len(performance))
    payload["factor_ic_rows"] = int(len(factor_ic))
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _report_pdf(summary: dict) -> bytes:
    lines = [
        "Research Report",
        f"run_id: {summary['run_id']}",
        f"data_end_date: {summary['data_end_date']}",
        f"feature_version: {summary['feature_version']}",
        f"model_version: {summary['model_version']}",
        f"parameter_version: {summary['parameter_version']}",
    ]
    metrics = summary.get("metrics", {})
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("metrics: n/a")
    content = "BT /F1 12 Tf 50 780 Td 15 TL " + " Tj T* ".join(
        f"({_pdf_escape(line)})" for line in lines
    )
    content += " Tj ET"
    content_bytes = content.encode("latin-1", errors="replace")
    stream = (
        b"<< /Length "
        + str(len(content_bytes)).encode()
        + b" >>\nstream\n"
        + content_bytes
        + b"\nendstream"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        stream,
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(pdf)
    pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode()
    pdf += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF".encode()
    )
    return bytes(pdf)


def _pdf_escape(text: object) -> str:
    return str(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
