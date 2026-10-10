import pytest
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler

from credit_risk.preprocess import CATEGORICAL, ORDINAL
from credit_risk.samplers import SAMPLER_NAMES, get_sampler


@pytest.mark.parametrize(
    ("name", "cls"), [("undersample", RandomUnderSampler), ("smotenc", SMOTENC)]
)
def test_each_name_builds_its_sampler_with_the_seed(name, cls):
    sampler = get_sampler(name, seed=7)
    assert isinstance(sampler, cls)
    assert sampler.random_state == 7


def test_names_cover_the_registry():
    assert SAMPLER_NAMES == ("undersample", "smotenc")


def test_smotenc_treats_categories_and_pay_codes_as_categorical():
    assert get_sampler("smotenc").categorical_features == CATEGORICAL + ORDINAL


def test_each_call_returns_a_new_object():
    # A shared object would carry fitted state from one experiment into the next.
    assert get_sampler("smotenc") is not get_sampler("smotenc")


def test_unknown_sampler_lists_valid_names():
    with pytest.raises(ValueError, match="undersample, smotenc"):
        get_sampler("smote")
