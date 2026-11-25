from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    parser.addoption("--run-cuda", action="store_true", help="Allow expensive generation tests on the VM")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-cuda"):
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="GPU runs require --run-cuda on the VM"))
