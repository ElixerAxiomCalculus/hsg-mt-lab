import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from typing import Any

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table


def generate_pdf(experiment: dict[str, Any], metrics: dict[str, Any]) -> bytes:
    target = io.BytesIO()
    document = SimpleDocTemplate(target, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm)
    styles = getSampleStyleSheet()
    story: list[Any] = [
        Paragraph("HSG-MT Lab — Research Report", styles["Title"]),
        Spacer(1, 8 * mm),
        Paragraph("PAPER TRADING / RESEARCH ONLY", styles["Heading2"]),
        Paragraph(f"Experiment: {experiment.get('name', 'Unnamed')}", styles["BodyText"]),
        Paragraph(f"Experiment ID: {experiment.get('id', 'N/A')}", styles["BodyText"]),
        Paragraph(f"Generated: {datetime.now(UTC).isoformat()}", styles["BodyText"]),
        Spacer(1, 6 * mm),
        Paragraph("Executive summary", styles["Heading1"]),
        Table([[key.replace("_", " ").title(), str(value)] for key, value in metrics.items()]),
        Spacer(1, 6 * mm),
        Paragraph("Limitations", styles["Heading1"]),
        Paragraph(
            "Paper trading only. Yahoo/yfinance data may be delayed or incomplete. "
            "Execution, slippage, and transaction costs are simulated. Short experiments are "
            "not statistically conclusive. Model uncertainty remains and no future "
            "profitability is claimed.",
            styles["BodyText"],
        ),
    ]
    document.build(story)
    return target.getvalue()


def _csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    target = io.StringIO()
    if rows:
        writer = csv.DictWriter(target, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    else:
        target.write("# No observations were available\n")
    return target.getvalue().encode()


def generate_research_archive(payload: dict[str, Any]) -> bytes:
    files: dict[str, bytes] = {
        "report.pdf": payload.get("report_pdf", b""),
        "experiment.json": json.dumps(
            payload.get("experiment", {}), default=str, indent=2
        ).encode(),
        "configuration.json": json.dumps(payload.get("configuration", {}), indent=2).encode(),
        "dataset_manifest.json": json.dumps(payload.get("dataset_manifest", {}), indent=2).encode(),
        "trades.csv": _csv_bytes(payload.get("trades", [])),
        "orders.csv": _csv_bytes(payload.get("orders", [])),
        "predictions.csv": _csv_bytes(payload.get("predictions", [])),
        "scanner_snapshots.csv": _csv_bytes(payload.get("scanner_snapshots", [])),
        "portfolio_equity.csv": _csv_bytes(payload.get("portfolio_equity", [])),
        "financial_metrics.json": json.dumps(
            payload.get("financial_metrics", {}), indent=2
        ).encode(),
        "model_metrics.json": json.dumps(payload.get("model_metrics", {}), indent=2).encode(),
        "data_quality_events.csv": _csv_bytes(payload.get("data_quality_events", [])),
        "agent_events.jsonl": b"\n".join(
            json.dumps(item, default=str).encode() for item in payload.get("agent_events", [])
        ),
        "model_manifest.json": json.dumps(payload.get("model_manifest", {}), indent=2).encode(),
        "README.txt": (
            b"HSG-MT Lab research archive. Missing observations are represented by empty "
            b"files. Paper trading only.\n"
        ),
    }
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    files["manifest.json"] = json.dumps(manifest, indent=2).encode()
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return target.getvalue()
