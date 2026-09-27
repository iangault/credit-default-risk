import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_eda_setup_cell_keeps_its_imports():
    # `make format` runs ruff --fix; unused-import removal must not strip the setup cell.
    # Checks names, not layout, so the author can reformat or add imports freely.
    nb = json.loads((ROOT / "notebooks" / "01-eda.ipynb").read_text())
    setup = "".join(nb["cells"][1]["source"])
    for name in ["altair as alt", "pandas as pd", "load_split", "TARGET", "PAY_STATUS"]:
        assert name in setup, f"{name!r} missing from the notebook's setup cell"
