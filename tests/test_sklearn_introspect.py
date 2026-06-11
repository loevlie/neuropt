"""Unit tests for sklearn-compatible model introspection."""

import json

import pytest

from neuropt.backends.base import BaseLLMBackend
from neuropt.introspect import (
    SKIP_PARAMS,
    _fallback_sklearn_search_space,
    build_sklearn_ml_context,
    build_sklearn_search_space_with_llm,
    introspect_sklearn,
    is_sklearn_compatible,
    make_sklearn_wrapped_train_fn,
)
from neuropt.search_space import Categorical, IntUniform, LogUniform, Uniform

sklearn = pytest.importorskip("sklearn")
from sklearn.ensemble import RandomForestClassifier  # noqa: E402


class StubBackend(BaseLLMBackend):
    """Returns a canned response and counts calls."""

    def __init__(self, response):
        super().__init__()
        self.response = response
        self.calls = 0

    def generate(self, prompt, max_tokens=1024):
        self.calls += 1
        return self.response

    def is_available(self):
        return True


@pytest.fixture
def rf_info():
    return introspect_sklearn(RandomForestClassifier(n_estimators=50, random_state=42))


class TestIntrospectSklearn:
    def test_is_sklearn_compatible(self):
        assert is_sklearn_compatible(RandomForestClassifier())
        assert not is_sklearn_compatible(object())

    def test_model_type(self, rf_info):
        assert rf_info["model_type"] == "RandomForestClassifier"
        assert "sklearn" in rf_info["model_module"]

    def test_skip_params_excluded(self, rf_info):
        for name in SKIP_PARAMS:
            assert name not in rf_info["tunable_params"]

    def test_none_params_included(self, rf_info):
        # max_depth=None by default — kept so the LLM can pick a range
        assert "max_depth" in rf_info["tunable_params"]
        assert rf_info["tunable_params"]["max_depth"] is None

    def test_current_values_captured(self, rf_info):
        assert rf_info["tunable_params"]["n_estimators"] == 50


class TestFallbackSklearnSearchSpace:
    def test_known_ranges_used(self, rf_info):
        space = _fallback_sklearn_search_space(rf_info)
        assert isinstance(space["min_samples_split"], IntUniform)
        assert isinstance(space["max_features"], Uniform)

    def test_bool_param_becomes_categorical(self):
        info = {"tunable_params": {"bootstrap": True}}
        space = _fallback_sklearn_search_space(info)
        assert isinstance(space["bootstrap"], Categorical)
        assert set(space["bootstrap"].choices) == {True, False}

    def test_int_param_gets_range_around_value(self):
        info = {"tunable_params": {"some_int": 30}}
        space = _fallback_sklearn_search_space(info)
        dim = space["some_int"]
        assert isinstance(dim, IntUniform)
        assert dim.low == 10 and dim.high == 90

    def test_reg_param_gets_log_range(self):
        info = {"tunable_params": {"reg_strength": 0.5}}
        space = _fallback_sklearn_search_space(info)
        assert isinstance(space["reg_strength"], LogUniform)


class TestLLMSklearnSearchSpace:
    def test_valid_response_builds_space(self, rf_info):
        backend = StubBackend(json.dumps({
            "n_estimators": {"type": "int", "min": 50, "max": 500},
            "max_features": {"type": "float", "min": 0.3, "max": 1.0},
            "min_samples_leaf": {"type": "int", "min": 1, "max": 20},
            "bootstrap": {"type": "bool"},
            "criterion": {"type": "choice", "choices": ["gini", "entropy"]},
        }))
        space = build_sklearn_search_space_with_llm(rf_info, backend)
        assert backend.calls == 1
        assert isinstance(space["n_estimators"], IntUniform)
        assert isinstance(space["max_features"], Uniform)
        assert isinstance(space["bootstrap"], Categorical)
        assert space["criterion"].choices == ["gini", "entropy"]

    def test_log_float_clamps_nonpositive_min(self, rf_info):
        backend = StubBackend(json.dumps({
            "ccp_alpha": {"type": "log_float", "min": 0, "max": 0.1},
        }))
        space = build_sklearn_search_space_with_llm(rf_info, backend)
        assert isinstance(space["ccp_alpha"], LogUniform)
        assert space["ccp_alpha"].low > 0

    def test_unknown_param_names_skipped(self, rf_info):
        backend = StubBackend(json.dumps({
            "not_a_real_param": {"type": "int", "min": 1, "max": 10},
            "n_estimators": {"type": "int", "min": 50, "max": 500},
        }))
        space = build_sklearn_search_space_with_llm(rf_info, backend)
        assert "not_a_real_param" not in space
        assert "n_estimators" in space

    def test_malformed_json_falls_back(self, rf_info):
        backend = StubBackend("sorry, I can't help with that")
        space = build_sklearn_search_space_with_llm(rf_info, backend)
        # Should be the heuristic fallback, not empty
        fallback = _fallback_sklearn_search_space(rf_info)
        assert {k: repr(v) for k, v in space.items()} == \
               {k: repr(v) for k, v in fallback.items()}

    def test_invalid_specs_skipped(self, rf_info):
        backend = StubBackend(json.dumps({
            "n_estimators": {"type": "int", "min": "low", "max": "high"},
            "max_features": {"type": "float", "min": 0.3, "max": 1.0},
        }))
        space = build_sklearn_search_space_with_llm(rf_info, backend)
        assert "n_estimators" not in space
        assert "max_features" in space


class TestSklearnWrappedTrainFn:
    def test_clones_and_sets_params(self):
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        received = {}

        def train_fn(config):
            received.update(config)
            return {"score": 0.5}

        wrapped = make_sklearn_wrapped_train_fn(model, train_fn)
        wrapped({"n_estimators": 99, "not_a_param": "extra"})

        cloned = received["model"]
        assert cloned is not model
        assert cloned.get_params()["n_estimators"] == 99
        # Original untouched
        assert model.get_params()["n_estimators"] == 10
        # Non-model keys still visible to train_fn
        assert received["not_a_param"] == "extra"


class TestSklearnMLContext:
    def test_context_lists_params(self, rf_info):
        space = _fallback_sklearn_search_space(rf_info)
        ctx = build_sklearn_ml_context(rf_info, space)
        assert "RandomForestClassifier" in ctx
        for name in space:
            assert name in ctx
