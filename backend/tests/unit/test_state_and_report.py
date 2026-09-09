import json
import zipfile
from io import BytesIO

import pytest

from app.schemas.experiment import ExperimentState, validate_transition
from app.services.reporting import generate_pdf, generate_research_archive


def test_state_transitions() -> None:
    validate_transition(ExperimentState.DRAFT, ExperimentState.INITIALIZING)
    with pytest.raises(ValueError, match="INVALID_STATE_TRANSITION"):
        validate_transition(ExperimentState.DRAFT, ExperimentState.COMPLETED)


def test_report_and_archive_generation() -> None:
    pdf = generate_pdf({"id": "exp-1", "name": "Test"}, {"total_return": "0.01"})
    assert pdf.startswith(b"%PDF")
    archive = generate_research_archive({"report_pdf": pdf, "experiment": {"id": "exp-1"}})
    with zipfile.ZipFile(BytesIO(archive)) as bundle:
        names = set(bundle.namelist())
        assert {"report.pdf", "manifest.json", "trades.csv", "agent_events.jsonl"} <= names
        manifest = json.loads(bundle.read("manifest.json"))
        assert "report.pdf" in manifest
