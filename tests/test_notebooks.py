import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_eda_setup_cell_keeps_its_imports():
    # `make format` runs ruff --fix; unused-import removal must not strip the scaffold.
    nb = json.loads((ROOT / "notebooks" / "01-eda.ipynb").read_text())
    setup = "".join(nb["cells"][1]["source"])
    assert "import altair as alt" in setup
    assert "import pandas as pd" in setup
    assert "BILL_AMTS, PAY_AMTS, PAY_STATUS, TARGET, load_split" in setup
