import pytest
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from credit_risk.models import MODEL_NAMES, get_model


def test_model_names():
    assert MODEL_NAMES == ("dummy", "logreg", "lgbm")


@pytest.mark.parametrize(
    ("name", "cls"),
    [("dummy", DummyClassifier), ("logreg", LogisticRegression), ("lgbm", LGBMClassifier)],
)
def test_get_model_returns_the_named_model(name, cls):
    assert isinstance(get_model(name), cls)


def test_defaults():
    assert get_model("dummy").strategy == "prior"
    assert get_model("logreg").max_iter == 1000
    assert get_model("lgbm").get_params()["verbose"] == -1


@pytest.mark.parametrize("name", ["dummy", "logreg", "lgbm"])
def test_seed_is_applied(name):
    assert get_model(name, seed=7).get_params()["random_state"] == 7


def test_params_override_defaults():
    model = get_model("logreg", {"C": 0.1, "max_iter": 50})
    assert model.C == 0.1
    assert model.max_iter == 50


def test_each_call_returns_a_new_model():
    assert get_model("logreg") is not get_model("logreg")


def test_unknown_name_lists_valid_names():
    with pytest.raises(ValueError) as exc:
        get_model("xgb")
    for name in MODEL_NAMES:
        assert name in str(exc.value)


def test_misspelled_param_raises():
    with pytest.raises((TypeError, ValueError)):
        get_model("logreg", {"CC": 1.0})
