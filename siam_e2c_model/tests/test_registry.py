import pytest

from siam_e2c_model.aggregation import (available_aggregations, create_aggregation,
                                        register_aggregation)


def test_builtins_registered():
    names = available_aggregations()
    assert names == tuple(sorted(names))
    for n in ("supervlad", "vlad", "residual"):
        assert n in names


def test_create_supervlad_needs_nothing():
    agg = create_aggregation("supervlad")
    assert agg.name == "supervlad"


def test_create_vlad_needs_centroids():
    import torch
    agg = create_aggregation("vlad", centroids=torch.randn(4, 8))
    assert agg.name == "vlad"
    with pytest.raises(ValueError):
        create_aggregation("vlad")                     # no centroids / blob
    with pytest.raises(ValueError):
        create_aggregation("residual", blob={})        # blob without 'centroids'


def test_unknown_lists_available():
    with pytest.raises(KeyError) as e:
        create_aggregation("nope")
    assert "supervlad" in str(e.value)


def test_duplicate_registration_forbidden():
    with pytest.raises(ValueError):
        register_aggregation("supervlad", lambda **kw: None)
