from pathlib import Path

import pytest

from app.services.hsg_adapter import AlgorithmError, AlgorithmStatus, HSGAdapter


def test_empty_algorithm_is_not_installed(tmp_path: Path) -> None:
    path = tmp_path / "algo.py"
    path.write_bytes(b"")
    adapter = HSGAdapter(path)
    assert adapter.validate_contract().status is AlgorithmStatus.NOT_INSTALLED
    assert not any(adapter.validate_contract().capabilities.values())


@pytest.mark.asyncio
async def test_empty_algorithm_never_predicts(tmp_path: Path) -> None:
    path = tmp_path / "algo.py"
    path.write_bytes(b"")
    with pytest.raises(AlgorithmError, match="not installed") as error:
        await HSGAdapter(path).predict(symbol="INFY.NS")
    assert error.value.code == "ALGORITHM_NOT_INSTALLED"


def test_invalid_contract_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "algo.py"
    path.write_text("def predict(**kwargs): return {}\n", encoding="utf-8")
    assert HSGAdapter(path).validate_contract().status is AlgorithmStatus.CONTRACT_INVALID


def test_dataclass_module_loads(tmp_path: Path) -> None:
    path = tmp_path / "algo.py"
    source = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class State:\n"
        "    version: str = 'test'\n"
        "def pretrain(**kwargs): return {'model_version': 'test'}\n"
        "def predict(**kwargs): return {}\n"
    )
    path.write_text(source, encoding="utf-8")
    status = HSGAdapter(path).validate_contract()
    assert status.status is AlgorithmStatus.INSTALLED
    assert status.capabilities["pretrain"] is True
    assert status.capabilities["predict"] is True
