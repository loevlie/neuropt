"""Core unit tests for neuropt — no LLM needed, all run with backend='none'."""

import copy
import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from neuropt import ArchSearch, Categorical, IntUniform, LogUniform, Uniform
from neuropt.backends.base import BaseLLMBackend
from neuropt.introspect import (
    AttentionPool2d,
    _classify_dropout_path,
    _detect_pretrained,
    _find_last_linear,
    _find_layer_groups,
    apply_config,
    build_ml_context,
    build_search_space,
    introspect,
    make_wrapped_train_fn,
)

# ── Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def simple_mlp():
    return nn.Sequential(
        nn.Linear(100, 50), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(50, 10),
    )


@pytest.fixture
def cnn_model():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.Mish(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(128, 10),
    )


@pytest.fixture
def transformer_model():
    return nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=64, nhead=4, dropout=0.1, batch_first=True),
        num_layers=2,
    )


@pytest.fixture
def tmp_log():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    yield path
    if os.path.exists(path):
        os.remove(path)


# ══════════════════════════════════════════════════════════════════════════
#  SEARCH SPACE
# ══════════════════════════════════════════════════════════════════════════

class TestSearchSpaceInference:
    def test_tuple_log_uniform(self):
        """lr-like names with wide range → LogUniform."""
        from neuropt.arch_search import _infer_dim
        dim = _infer_dim("lr", (1e-4, 1e-1))
        assert isinstance(dim, LogUniform)

    def test_tuple_int_uniform(self):
        from neuropt.arch_search import _infer_dim
        dim = _infer_dim("n_layers", (2, 8))
        assert isinstance(dim, IntUniform)

    def test_list_categorical(self):
        from neuropt.arch_search import _infer_dim
        dim = _infer_dim("activation", ["relu", "gelu"])
        assert isinstance(dim, Categorical)

    def test_passthrough_dimension(self):
        from neuropt.arch_search import _infer_dim
        orig = LogUniform(1e-4, 1e-1)
        dim = _infer_dim("x", orig)
        assert dim is orig

    def test_normalize_search_space(self):
        from neuropt.arch_search import _normalize_search_space
        space = _normalize_search_space({
            "lr": (1e-4, 1e-1),
            "n_layers": (2, 8),
            "act": ["relu", "gelu"],
        })
        assert isinstance(space["lr"], LogUniform)
        assert isinstance(space["n_layers"], IntUniform)
        assert isinstance(space["act"], Categorical)


# ══════════════════════════════════════════════════════════════════════════
#  INTROSPECTION
# ══════════════════════════════════════════════════════════════════════════

class TestIntrospect:
    def test_mlp_detection(self, simple_mlp):
        info = introspect(simple_mlp)
        assert info["has_linear"]
        assert info["has_dropout"]
        assert info["dropout_rate"] == 0.2
        assert len(info["dropout_paths"]) == 1
        assert "ReLU" in info["activations_found"]
        assert not info["has_batchnorm"]
        assert not info["has_layernorm"]
        assert not info["has_pool"]

    def test_cnn_detection(self, cnn_model):
        info = introspect(cnn_model)
        assert info["has_conv"]
        assert info["has_batchnorm"]
        assert info["has_pool"]
        assert info["pool_type"] == "avg"
        assert len(info["pool_paths"]) == 1
        assert "ReLU" in info["activations_found"]
        assert "Mish" in info["activations_found"]

    def test_transformer_detection(self, transformer_model):
        info = introspect(transformer_model)
        assert info["has_layernorm"]
        assert len(info["layernorm_paths"]) >= 2
        assert info["has_dropout"]
        assert len(info["mha_dropout_paths"]) == 2

    def test_all_dropout_types_detected(self):
        model = nn.Sequential(
            nn.Dropout(0.1), nn.Dropout1d(0.2), nn.Dropout2d(0.3),
            nn.AlphaDropout(0.4), nn.Linear(10, 5),
        )
        info = introspect(model)
        assert info["has_dropout"]
        assert len(info["dropout_paths"]) == 4
        # First rate kept
        assert info["dropout_rate"] == 0.1

    def test_all_activation_types_detected(self):
        model = nn.Sequential(
            nn.ReLU(), nn.GELU(), nn.SiLU(), nn.Mish(),
            nn.Hardswish(), nn.PReLU(), nn.LeakyReLU(),
            nn.Linear(10, 5),
        )
        info = introspect(model)
        assert len(info["activations_found"]) == 7
        assert len(info["activation_paths"]) == 7

    def test_pretrained_override_true(self, simple_mlp):
        info = introspect(simple_mlp, pretrained=True)
        assert info["is_pretrained"] is True

    def test_pretrained_override_false(self, simple_mlp):
        info = introspect(simple_mlp, pretrained=False)
        assert info["is_pretrained"] is False

    def test_pretrained_autodetect_random_init(self, simple_mlp):
        assert _detect_pretrained(simple_mlp) is False

    def test_pretrained_autodetect_trained(self, simple_mlp):
        # Scale weights down to simulate training
        for p in simple_mlp.parameters():
            if p.dim() >= 2:
                p.data *= 0.1
        assert _detect_pretrained(simple_mlp) is True

    def test_last_linear_found(self, cnn_model):
        path = _find_last_linear(cnn_model)
        assert path is not None
        assert "8" in path or "Linear" in str(type(cnn_model[8]))

    def test_layer_groups_transformer(self, transformer_model):
        groups = _find_layer_groups(transformer_model)
        assert len(groups) >= 1
        _, paths = groups[0]
        assert len(paths) >= 2


class TestDropoutClassification:
    def test_attn_path(self):
        assert _classify_dropout_path("encoder.self_attn.dropout") == "attn"
        assert _classify_dropout_path("layers.0.attention.drop") == "attn"

    def test_ff_path(self):
        assert _classify_dropout_path("encoder.ff.dropout") == "ff"
        assert _classify_dropout_path("blocks.0.mlp.drop") == "ff"

    def test_embed_path(self):
        assert _classify_dropout_path("embedding.dropout") == "embed"

    def test_default_path(self):
        assert _classify_dropout_path("layer.0.dropout1") == "default"


# ══════════════════════════════════════════════════════════════════════════
#  SEARCH SPACE GENERATION
# ══════════════════════════════════════════════════════════════════════════

class TestBuildSearchSpace:
    def test_mlp_space(self, simple_mlp):
        info = introspect(simple_mlp)
        space = build_search_space(info)
        assert "activation" in space
        assert "dropout" in space
        assert "lr" in space
        assert "wd" in space
        assert "optimizer" in space
        # No batchnorm or layernorm
        assert "use_batchnorm" not in space
        assert "use_layernorm" not in space

    def test_cnn_space(self, cnn_model):
        info = introspect(cnn_model)
        space = build_search_space(info)
        assert "activation" in space
        assert "use_batchnorm" in space
        assert "pool_type" in space
        assert "avg" in space["pool_type"].choices
        assert "max" in space["pool_type"].choices
        assert "attention" in space["pool_type"].choices

    def test_transformer_space(self, transformer_model):
        info = introspect(transformer_model)
        space = build_search_space(info)
        assert "use_layernorm" in space
        assert "mha_dropout" in space

    def test_pretrained_adds_finetune_params(self, simple_mlp):
        info = introspect(simple_mlp, pretrained=True)
        space = build_search_space(info)
        assert "freeze_strategy" in space
        assert "lr_layer_decay" in space
        assert "l2sp_regularization" in space

    def test_activation_choices_complete(self, simple_mlp):
        info = introspect(simple_mlp)
        space = build_search_space(info)
        choices = space["activation"].choices
        assert len(choices) == 7
        for name in ["relu", "gelu", "silu", "leaky_relu", "mish", "hardswish", "prelu"]:
            assert name in choices


# ══════════════════════════════════════════════════════════════════════════
#  APPLY CONFIG
# ══════════════════════════════════════════════════════════════════════════

class TestApplyConfig:
    def test_swap_activation(self, simple_mlp):
        info = introspect(simple_mlp)
        m = copy.deepcopy(simple_mlp)
        apply_config(m, {"activation": "gelu"}, info)
        assert isinstance(m[1], nn.GELU)

    def test_swap_to_mish(self, simple_mlp):
        info = introspect(simple_mlp)
        m = copy.deepcopy(simple_mlp)
        apply_config(m, {"activation": "mish"}, info)
        assert isinstance(m[1], nn.Mish)

    def test_swap_to_prelu(self, simple_mlp):
        info = introspect(simple_mlp)
        m = copy.deepcopy(simple_mlp)
        apply_config(m, {"activation": "prelu"}, info)
        assert isinstance(m[1], nn.PReLU)

    def test_set_dropout(self, simple_mlp):
        info = introspect(simple_mlp)
        m = copy.deepcopy(simple_mlp)
        apply_config(m, {"dropout": 0.42}, info)
        assert m[2].p == 0.42

    def test_toggle_batchnorm_off(self, cnn_model):
        info = introspect(cnn_model)
        m = copy.deepcopy(cnn_model)
        apply_config(m, {"use_batchnorm": False}, info)
        for path in info["batchnorm_paths"]:
            parts = path.split(".")
            mod = m
            for p in parts:
                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
            assert isinstance(mod, nn.Identity)

    def test_toggle_layernorm_off(self, transformer_model):
        info = introspect(transformer_model)
        m = copy.deepcopy(transformer_model)
        apply_config(m, {"use_layernorm": False}, info)
        for path in info["layernorm_paths"]:
            parts = path.split(".")
            mod = m
            for p in parts:
                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
            assert isinstance(mod, nn.Identity)

    def test_swap_pool_to_max(self, cnn_model):
        info = introspect(cnn_model)
        pool_idx = int(info["pool_paths"][0])
        m = copy.deepcopy(cnn_model)
        apply_config(m, {"pool_type": "max"}, info)
        assert isinstance(m[pool_idx], nn.AdaptiveMaxPool2d)

    def test_swap_pool_to_attention(self, cnn_model):
        info = introspect(cnn_model)
        pool_idx = int(info["pool_paths"][0])
        m = copy.deepcopy(cnn_model)
        apply_config(m, {"pool_type": "attention"}, info)
        assert "AttentionPool2d" in type(m[pool_idx]).__name__
        # Verify full forward pass works
        x = torch.randn(2, 3, 32, 32)
        out = m(x)
        assert out.shape == (2, 10)

    def test_set_mha_dropout(self, transformer_model):
        info = introspect(transformer_model)
        m = copy.deepcopy(transformer_model)
        apply_config(m, {"mha_dropout": 0.25}, info)
        for path in info["mha_dropout_paths"]:
            parts = path.split(".")
            mod = m
            for p in parts:
                mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
            assert mod.dropout == 0.25

    def test_freeze_head_only(self, simple_mlp):
        info = introspect(simple_mlp, pretrained=True)
        m = copy.deepcopy(simple_mlp)
        apply_config(m, {"freeze_strategy": "head_only"}, info)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        # Only the last linear's params should be trainable
        assert len(trainable) == 2  # weight + bias

    def test_freeze_full(self, simple_mlp):
        info = introspect(simple_mlp, pretrained=True)
        m = copy.deepcopy(simple_mlp)
        apply_config(m, {"freeze_strategy": "full"}, info)
        trainable = [n for n, p in m.named_parameters() if p.requires_grad]
        assert len(trainable) == sum(1 for _ in m.parameters())  # all trainable


# ══════════════════════════════════════════════════════════════════════════
#  WRAPPED TRAIN FN
# ══════════════════════════════════════════════════════════════════════════

class TestWrappedTrainFn:
    def test_model_is_deep_copy(self, simple_mlp):
        info = introspect(simple_mlp)
        received = {}

        def train_fn(config):
            received["model"] = config["model"]
            return {"score": 0.5}

        wrapped = make_wrapped_train_fn(simple_mlp, train_fn, info)
        wrapped({"activation": "gelu", "dropout": 0.1})
        assert received["model"] is not simple_mlp

    def test_l2sp_injects_weights(self, simple_mlp):
        info = introspect(simple_mlp, pretrained=True)
        received = {}

        def train_fn(config):
            received.update(config)
            return {"score": 0.5}

        wrapped = make_wrapped_train_fn(simple_mlp, train_fn, info)
        wrapped({"l2sp_regularization": True, "freeze_strategy": "full"})
        assert "pretrained_weights" in received

    def test_l2sp_not_injected_when_off(self, simple_mlp):
        info = introspect(simple_mlp, pretrained=True)
        received = {}

        def train_fn(config):
            received.update(config)
            return {"score": 0.5}

        wrapped = make_wrapped_train_fn(simple_mlp, train_fn, info)
        wrapped({"l2sp_regularization": False, "freeze_strategy": "full"})
        assert "pretrained_weights" not in received


# ══════════════════════════════════════════════════════════════════════════
#  ML CONTEXT
# ══════════════════════════════════════════════════════════════════════════

class TestBuildMLContext:
    def test_mentions_activations(self, simple_mlp):
        info = introspect(simple_mlp)
        ctx = build_ml_context(info)
        assert "ReLU" in ctx

    def test_mentions_layernorm(self, transformer_model):
        info = introspect(transformer_model)
        ctx = build_ml_context(info)
        assert "Layer normalization" in ctx

    def test_mentions_pooling(self, cnn_model):
        info = introspect(cnn_model)
        ctx = build_ml_context(info)
        assert "Adaptive pooling" in ctx

    def test_pretrained_guidance(self, simple_mlp):
        info = introspect(simple_mlp, pretrained=True)
        ctx = build_ml_context(info)
        assert "Fine-tuning" in ctx or "fine-tuning" in ctx
        assert "head_only" in ctx
        assert "gradual_unfreeze" in ctx


# ══════════════════════════════════════════════════════════════════════════
#  ARCH SEARCH (end-to-end with random backend)
# ══════════════════════════════════════════════════════════════════════════

class TestArchSearch:
    def test_basic_run(self, tmp_log):
        search = ArchSearch(
            train_fn=lambda cfg: {"score": cfg["x"]},
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=3)
        assert search.total_experiments == 3
        assert search.best_score <= 10.0

    def test_minimize_true(self, tmp_log):
        search = ArchSearch(
            train_fn=lambda cfg: {"score": cfg["x"]},
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
            minimize=True,
        )
        search.run(max_evals=5)
        assert search.best_score == min(
            cfg["x"] for cfg in [search.best_config]  # at least works
        )

    def test_minimize_false(self, tmp_log):
        search = ArchSearch(
            train_fn=lambda cfg: {"score": cfg["x"]},
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
            minimize=False,
        )
        search.run(max_evals=5)
        assert search.best_score >= 1.0

    def test_returns_scalar(self, tmp_log):
        """train_fn can return just a number."""
        search = ArchSearch(
            train_fn=lambda cfg: cfg["x"] * 2,
            search_space={"x": (1.0, 5.0)},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=3)
        assert search.total_experiments == 3

    def test_custom_return_keys_logged(self, tmp_log):
        """Extra keys in train_fn return dict should be logged."""
        def train_fn(config):
            return {
                "score": config["x"],
                "f1": 0.82,
                "n_train": 5000,
                "train_losses": [1.0, 0.5],
                "custom_curve": [0.3, 0.6, 0.9],
            }

        search = ArchSearch(
            train_fn=train_fn,
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=2)

        with open(tmp_log) as f:
            row = json.loads(f.readline())

        assert "f1" in row["scalars"]
        assert "n_train" in row["scalars"]
        assert "train_losses" in row["curves"]
        assert "custom_curve" in row["curves"]

    def test_log_resume(self, tmp_log):
        """Run should resume from existing log."""
        search1 = ArchSearch(
            train_fn=lambda cfg: {"score": cfg["x"]},
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
        )
        search1.run(max_evals=3)

        search2 = ArchSearch(
            train_fn=lambda cfg: {"score": cfg["x"]},
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
        )
        search2.run(max_evals=3)
        assert search2.total_experiments == 6

    def test_nan_score_handled(self, tmp_log):
        search = ArchSearch(
            train_fn=lambda cfg: {"score": float("nan")},
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=2)
        assert search.total_experiments == 2

    def test_error_in_train_fn_handled(self, tmp_log):
        call_count = [0]

        def bad_fn(config):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("oops")
            return {"score": 0.5}

        search = ArchSearch(
            train_fn=bad_fn,
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=3)
        assert search.total_experiments == 3


# ══════════════════════════════════════════════════════════════════════════
#  FROM_MODEL
# ══════════════════════════════════════════════════════════════════════════

class TestFromModel:
    def test_from_pytorch_model(self, simple_mlp, tmp_log):
        search = ArchSearch.from_model(
            simple_mlp,
            lambda cfg: {"score": 0.5},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=2)
        assert search.total_experiments == 2

    def test_from_model_pretrained_kwarg(self, simple_mlp, tmp_log):
        search = ArchSearch.from_model(
            simple_mlp,
            lambda cfg: {"score": 0.5},
            backend="none",
            pretrained=True,
            log_path=tmp_log,
        )
        assert "freeze_strategy" in search.search_space

    def test_from_sklearn_model(self, tmp_log):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        search = ArchSearch.from_model(
            model,
            lambda cfg: {"score": 0.5},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=2)
        assert search.total_experiments == 2

    def test_from_xgboost_model(self, tmp_log):
        xgb = pytest.importorskip("xgboost")
        model = xgb.XGBClassifier(n_estimators=10, verbosity=0)

        search = ArchSearch.from_model(
            model,
            lambda cfg: {"score": 0.5},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=2)
        assert search.total_experiments == 2


# ══════════════════════════════════════════════════════════════════════════
#  ATTENTION POOL
# ══════════════════════════════════════════════════════════════════════════

class TestAttentionPool:
    def test_forward_pass(self):
        pool_cls = AttentionPool2d.get_cls()
        pool = pool_cls(channels=64)
        x = torch.randn(2, 64, 8, 8)
        out = pool(x)
        assert out.shape == (2, 64, 1, 1)

    def test_class_name_no_underscore(self):
        pool_cls = AttentionPool2d.get_cls()
        pool = pool_cls(channels=32)
        assert not type(pool).__name__.startswith("_")


# ══════════════════════════════════════════════════════════════════════════
#  BACKWARD COMPAT (old log format)
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
#  API COST TRACKING
# ══════════════════════════════════════════════════════════════════════════

class TestCostTracking:
    def test_base_backend_initializes_counters(self):
        class DummyBackend(BaseLLMBackend):
            def generate(self, prompt, max_tokens=1024):
                return ""
            def is_available(self):
                return True

        b = DummyBackend()
        assert b.total_input_tokens == 0
        assert b.total_output_tokens == 0
        assert b.total_cost is None

    def test_claude_cost_calculation(self):
        from neuropt.backends.claude_backend import ClaudeBackend
        b = ClaudeBackend.__new__(ClaudeBackend)
        b.total_input_tokens = 0
        b.total_output_tokens = 0
        b._model = "claude-haiku-4-5-20251001"
        b._client = None
        # Simulate usage
        b.total_input_tokens = 10_000
        b.total_output_tokens = 2_000
        # Haiku: $1/M in, $5/M out
        expected = 10_000 * 1.0 / 1e6 + 2_000 * 5.0 / 1e6
        assert abs(b.total_cost - expected) < 1e-9

    def test_claude_cost_family_fallback(self):
        from neuropt.backends.claude_backend import ClaudeBackend
        b = ClaudeBackend.__new__(ClaudeBackend)
        b.total_input_tokens = 1_000_000
        b.total_output_tokens = 100_000
        b._model = "claude-sonnet-4-5-some-future-version"
        b._client = None
        # Should match "claude-sonnet" family: $3/M in, $15/M out
        expected = 1_000_000 * 3.0 / 1e6 + 100_000 * 15.0 / 1e6
        assert abs(b.total_cost - expected) < 1e-9

    def test_claude_cost_unknown_model(self):
        from neuropt.backends.claude_backend import ClaudeBackend
        b = ClaudeBackend.__new__(ClaudeBackend)
        b.total_input_tokens = 1000
        b.total_output_tokens = 500
        b._model = "some-unknown-model"
        b._client = None
        assert b.total_cost is None

    def test_summary_prints_cost(self, tmp_log, capsys):
        search = ArchSearch(
            train_fn=lambda cfg: {"score": cfg["x"]},
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=2)
        captured = capsys.readouterr().out
        # backend=none has no tokens, so no cost line
        assert "API cost" not in captured


class TestBackwardCompat:
    def test_old_log_format_resume(self, tmp_log):
        """Old logs with val_loss/val_accuracy should still work."""
        old_rows = [
            {"id": 1, "config": {"x": 5.0}, "val_loss": 0.5, "val_accuracy": 0.9,
             "train_losses": [1.0, 0.5], "val_losses": [1.1, 0.6],
             "status": "ok"},
        ]
        with open(tmp_log, "w") as f:
            for row in old_rows:
                f.write(json.dumps(row) + "\n")

        search = ArchSearch(
            train_fn=lambda cfg: {"score": cfg["x"]},
            search_space={"x": (1.0, 10.0)},
            backend="none",
            log_path=tmp_log,
        )
        search.run(max_evals=2)
        assert search.total_experiments == 3  # 1 old + 2 new


# ══════════════════════════════════════════════════════════════════════════
#  DIMENSION VALIDATION
# ══════════════════════════════════════════════════════════════════════════

class TestDimensionValidation:
    def test_uniform_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            Uniform(2.0, 1.0)

    def test_loguniform_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            LogUniform(1.0, 0.5)

    def test_loguniform_rejects_nonpositive_low(self):
        with pytest.raises(ValueError):
            LogUniform(0, 1.0)

    def test_intuniform_rejects_inverted_bounds(self):
        with pytest.raises(ValueError):
            IntUniform(5, 3)

    def test_categorical_rejects_empty(self):
        with pytest.raises(ValueError):
            Categorical([])


# ══════════════════════════════════════════════════════════════════════════
#  DIMENSION INFERENCE PRECEDENCE
# ══════════════════════════════════════════════════════════════════════════

class TestInferDimPrecedence:
    """Pin _infer_dim's precedence: int-like name > log heuristic > int bounds."""

    def test_int_name_wins_over_large_ratio(self):
        from neuropt.arch_search import _infer_dim
        dim = _infer_dim("hidden_dim", (32, 4096))
        assert isinstance(dim, IntUniform)

    def test_large_ratio_int_tuple_becomes_log(self):
        from neuropt.arch_search import _infer_dim
        dim = _infer_dim("foo", (1, 1000))
        assert isinstance(dim, LogUniform)

    def test_plain_int_tuple_becomes_int(self):
        from neuropt.arch_search import _infer_dim
        dim = _infer_dim("foo", (1, 50))
        assert isinstance(dim, IntUniform)

    def test_log_scale_name(self):
        from neuropt.arch_search import _infer_dim
        dim = _infer_dim("lr", (1e-4, 1e-1))
        assert isinstance(dim, LogUniform)

    def test_plain_float_tuple_becomes_uniform(self):
        from neuropt.arch_search import _infer_dim
        dim = _infer_dim("x", (0.1, 0.9))
        assert isinstance(dim, Uniform)


# ══════════════════════════════════════════════════════════════════════════
#  TOKEN TRACKING THROUGH THE SEARCH SUMMARY
# ══════════════════════════════════════════════════════════════════════════

class TestTokenTracking:
    def test_backend_tokens_reach_summary(self, tmp_log, capsys):
        class CountingBackend(BaseLLMBackend):
            def generate(self, prompt, max_tokens=1024):
                self.total_input_tokens += 100
                self.total_output_tokens += 50
                return '[{"x": 2.0}, {"x": 3.0}, {"x": 4.0}]'

            def is_available(self):
                return True

        search = ArchSearch(
            train_fn=lambda cfg: {"score": cfg["x"]},
            search_space={"x": (1.0, 10.0)},
            backend=CountingBackend(),
            log_path=tmp_log,
        )
        search.run(max_evals=3)
        captured = capsys.readouterr().out
        assert "Tokens: 100 in + 50 out" in captured

    def test_openai_cost_calculation(self):
        from neuropt.backends.openai_backend import OpenAIBackend
        b = OpenAIBackend.__new__(OpenAIBackend)
        b.total_input_tokens = 10_000
        b.total_output_tokens = 2_000
        b._model = "gpt-4o-mini"
        b._client = None
        # gpt-4o-mini: $0.15/M in, $0.60/M out
        expected = 10_000 * 0.15 / 1e6 + 2_000 * 0.60 / 1e6
        assert abs(b.total_cost - expected) < 1e-9

    def test_openai_cost_dated_snapshot_prefix(self):
        from neuropt.backends.openai_backend import OpenAIBackend
        b = OpenAIBackend.__new__(OpenAIBackend)
        b.total_input_tokens = 1_000_000
        b.total_output_tokens = 0
        b._model = "gpt-4o-mini-2024-07-18"
        b._client = None
        # Must match gpt-4o-mini ($0.15/M), not the shorter gpt-4o prefix
        assert abs(b.total_cost - 0.15) < 1e-9

    def test_openai_cost_unknown_model(self):
        from neuropt.backends.openai_backend import OpenAIBackend
        b = OpenAIBackend.__new__(OpenAIBackend)
        b.total_input_tokens = 1000
        b.total_output_tokens = 500
        b._model = "some-unknown-model"
        b._client = None
        assert b.total_cost is None


# ══════════════════════════════════════════════════════════════════════════
#  POOL DIMENSIONALITY
# ══════════════════════════════════════════════════════════════════════════

class TestPoolDims:
    def test_2d_pool_offers_attention(self):
        model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.AdaptiveAvgPool2d(1))
        info = introspect(model)
        space = build_search_space(info)
        assert "attention" in space["pool_type"].choices

    def test_1d_pool_excludes_attention(self):
        model = nn.Sequential(nn.Conv1d(3, 8, 3), nn.AdaptiveAvgPool1d(1))
        info = introspect(model)
        space = build_search_space(info)
        assert info["pool_dims"] == {"1d"}
        assert "attention" not in space["pool_type"].choices
        assert {"avg", "max"} <= set(space["pool_type"].choices)


# ══════════════════════════════════════════════════════════════════════════
#  CLI RESULTS (min/max direction)
# ══════════════════════════════════════════════════════════════════════════

class TestCLIResults:
    @pytest.fixture
    def maximize_log(self, tmp_log):
        rows = [
            {"id": 1, "config": {"x": 1.0}, "score": 0.6,
             "scalars": {"accuracy": 0.6}, "status": "ok"},
            {"id": 2, "config": {"x": 2.0}, "score": 0.9,
             "scalars": {"accuracy": 0.9}, "status": "ok"},
            {"id": 3, "config": {"x": 3.0}, "score": 0.7,
             "scalars": {"accuracy": 0.7}, "status": "ok"},
        ]
        with open(tmp_log, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return tmp_log

    def test_results_minimize_default(self, maximize_log):
        from typer.testing import CliRunner

        from neuropt.cli import app
        result = CliRunner().invoke(app, ["results", maximize_log])
        assert result.exit_code == 0
        # Lowest score listed first
        assert result.output.index("score=0.6000") < result.output.index("score=0.9000")

    def test_results_maximize_flag(self, maximize_log):
        from typer.testing import CliRunner

        from neuropt.cli import app
        result = CliRunner().invoke(app, ["results", maximize_log, "--maximize"])
        assert result.exit_code == 0
        # Highest score listed first
        assert result.output.index("score=0.9000") < result.output.index("score=0.6000")


# ══════════════════════════════════════════════════════════════════════════
#  REGRESSIONS FROM CODE REVIEW
# ══════════════════════════════════════════════════════════════════════════

class _CannedLLM(BaseLLMBackend):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)

    def generate(self, prompt, max_tokens=1024):
        self.total_input_tokens += 10
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]

    def is_available(self):
        return True


class TestRunLifecycle:
    def test_run_again_after_interrupt(self, tmp_log):
        search = ArchSearch(lambda cfg: {"score": cfg["x"]}, {"x": (1.0, 10.0)},
                            backend="none", log_path=tmp_log)
        search._shutdown = True  # as if Ctrl+C stopped a previous run()
        search.run(max_evals=2)
        assert search.total_experiments == 2

    def test_signal_handlers_restored(self, tmp_log):
        import signal
        before = signal.getsignal(signal.SIGINT)
        search = ArchSearch(lambda cfg: {"score": cfg["x"]}, {"x": (1.0, 10.0)},
                            backend="none", log_path=tmp_log)
        search.run(max_evals=1)
        assert signal.getsignal(signal.SIGINT) is before

    def test_run_from_worker_thread(self, tmp_log, capsys):
        import threading
        search = ArchSearch(lambda cfg: {"score": cfg["x"]}, {"x": (1.0, 10.0)},
                            backend="none", log_path=tmp_log)
        errors = []

        def target():
            try:
                search.run(max_evals=2)
            except Exception as e:  # pragma: no cover - the bug this guards against
                errors.append(e)

        t = threading.Thread(target=target)
        t.start()
        t.join()
        assert not errors
        assert search.total_experiments == 2
        rows = [json.loads(line) for line in open(tmp_log)]
        assert all(r["status"] == "ok" for r in rows)
        assert "timeout disabled" in capsys.readouterr().out


class TestTruncatedLogResume:
    def test_partial_last_line_is_skipped(self, tmp_log, capsys):
        with open(tmp_log, "w") as f:
            f.write(json.dumps({"id": 1, "config": {"x": 5.0}, "score": 5.0, "status": "ok"}) + "\n")
            f.write(json.dumps({"id": 2, "config": {"x": 3.0}, "score": 3.0, "status": "ok"}) + "\n")
            f.write('{"id": 3, "config": {"x": 2.0}, "sco')  # crash mid-write
        search = ArchSearch(lambda cfg: {"score": cfg["x"]}, {"x": (1.0, 10.0)},
                            backend="none", log_path=tmp_log)
        search.run(max_evals=2)
        assert search.total_experiments == 4
        assert search.best_score <= 3.0
        out = capsys.readouterr().out
        assert "skipped 1 unreadable line" in out
        # The new rows start on their own line and parse cleanly
        lines = open(tmp_log).read().splitlines()
        parsed = []
        for line in lines:
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        assert len(parsed) == 4
        assert [r["id"] for r in parsed] == [1, 2, 3, 4]


class TestClaudePricing:
    @pytest.mark.parametrize("model, prices", [
        ("claude-opus-4-6", (5.0, 25.0)),
        ("claude-opus-5-5", (4.0, 20.0)),
        ("claude-opus-5", (5.0, 25.0)),
        ("claude-sonnet-4-5-20250929", (3.0, 15.0)),
        ("claude-sonnet-5-5", (2.0, 10.0)),
        ("claude-haiku-5-5", (0.10, 0.50)),
        ("claude-haiku-4-5-20251001", (1.0, 5.0)),
        ("claude-fable-5-1", (10.0, 50.0)),
    ])
    def test_prices(self, model, prices):
        from neuropt.backends.claude_backend import ClaudeBackend
        b = ClaudeBackend(model=model)
        b.total_input_tokens = 1_000_000
        b.total_output_tokens = 1_000_000
        assert b.total_cost == pytest.approx(sum(prices))

    def test_unversioned_family_is_unknown(self):
        from neuropt.backends.claude_backend import ClaudeBackend
        b = ClaudeBackend(model="claude-opus")
        b.total_input_tokens = 1000
        assert b.total_cost is None

    def test_generate_skips_non_text_blocks(self):
        from types import SimpleNamespace

        from neuropt.backends.claude_backend import ClaudeBackend
        message = SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""),
                     SimpleNamespace(type="text", text="[1]")],
            usage=SimpleNamespace(input_tokens=3, output_tokens=4))
        client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: message))
        b = ClaudeBackend()
        b._client = client
        assert b.generate("hi") == "[1]"


class TestOpenAIParams:
    def test_uses_max_completion_tokens(self):
        from types import SimpleNamespace

        from neuropt.backends.openai_backend import OpenAIBackend
        seen = {}

        def create(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(
                usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

        b = OpenAIBackend(model="gpt-5")
        b._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        b.generate("hi", max_tokens=77)
        assert seen["max_completion_tokens"] == 77
        assert "max_tokens" not in seen


class TestConfigKey:
    def test_tiny_log_scale_values_stay_distinct(self):
        from neuropt.arch_search import _config_key
        assert _config_key({"wd": 1e-7}) != _config_key({"wd": 5e-7})
        assert _config_key({"wd": 1.2e-5}) != _config_key({"wd": 1.24e-5})

    def test_float_noise_still_matches(self):
        from neuropt.arch_search import _config_key
        assert _config_key({"lr": 0.0123}) == _config_key({"lr": 0.012299999999999997})


class TestResponseParsing:
    def _search(self, tmp_log, batch_size=3):
        return ArchSearch(lambda cfg: {"score": cfg["x"]}, {"x": (1.0, 10.0)},
                          backend="none", log_path=tmp_log, batch_size=batch_size)

    def test_trailing_prose_with_brackets(self, tmp_log):
        search = self._search(tmp_log)
        response = ('Here you go: [{"x": 2}, {"x": 3}, {"x": 4}]\n'
                    "Note: I varied [x] only.")
        configs = search._parse_response(response)
        assert [c["x"] for c in configs] == [2.0, 3.0, 4.0]

    def test_extra_configs_truncated(self, tmp_log):
        search = self._search(tmp_log, batch_size=2)
        configs = search._parse_response('[{"x": 2}, {"x": 3}, {"x": 4}]')
        assert [c["x"] for c in configs] == [2.0, 3.0]

    def test_short_answer_topped_up(self, tmp_log):
        import random as _random
        search = ArchSearch(lambda cfg: {"score": cfg["x"]}, {"x": (1.0, 10.0)},
                            backend=_CannedLLM(['[{"x": 2}]']), log_path=tmp_log)
        configs, source = search._get_configs([], _random.Random(0))
        assert source == "llm"
        assert len(configs) == 3
        assert configs[0]["x"] == 2.0

    def test_no_array_returns_none(self, tmp_log):
        assert self._search(tmp_log)._parse_response("no idea, sorry") is None
        assert self._search(tmp_log)._parse_response("[]") is None

    def test_sklearn_ranges_with_surrounding_prose(self):
        from neuropt.introspect import build_sklearn_search_space_with_llm
        info = {"model_type": "M", "tunable_params": {"max_depth": 5, "alpha": 0.1}}
        backend = _CannedLLM(['Sure! {"max_depth": {"type": "int", "min": 2, "max": 9}, '
                              '"alpha": "oops"} Hope that {helps}.'])
        space = build_sklearn_search_space_with_llm(info, backend)
        assert isinstance(space["max_depth"], IntUniform)
        assert "alpha" not in space  # malformed spec skipped instead of crashing


class TestCategoricalValidation:
    def test_numeric_choice_returns_choice_object(self):
        from neuropt.arch_search import _validate_config
        cleaned = _validate_config({"bs": 64.0}, {"bs": Categorical([32, 64, 128])})
        assert cleaned == {"bs": 64}
        assert type(cleaned["bs"]) is int

    def test_bool_not_accepted_for_int_choices(self):
        from neuropt.arch_search import _validate_config
        assert _validate_config({"flag": True}, {"flag": Categorical([0, 1])}) is None

    def test_bool_choices_still_coerce(self):
        from neuropt.arch_search import _validate_config
        space = {"bn": Categorical([True, False])}
        assert _validate_config({"bn": "true"}, space) == {"bn": True}
        assert _validate_config({"bn": 0}, space) == {"bn": False}


class TestSklearnBackendResolvedOnce:
    def test_range_tokens_reach_search_backend(self, tmp_log, monkeypatch):
        pytest.importorskip("sklearn")
        from sklearn.ensemble import RandomForestClassifier

        import neuropt.backends as backends
        made = []

        def factory():
            made.append(_CannedLLM(['{"max_depth": {"type": "int", "min": 2, "max": 9}}']))
            return made[-1]

        monkeypatch.setitem(backends._REGISTRY, "claude", factory)
        search = ArchSearch.from_model(RandomForestClassifier(n_estimators=5),
                                       lambda cfg: {"score": 0.5},
                                       backend="claude", log_path=tmp_log)
        assert len(made) == 1  # one backend instance, not one per resolution
        assert search._backend is made[0]
        assert search._backend.total_input_tokens == 10  # the range request is counted


class TestPromptDetails:
    def _search(self, tmp_log, **kw):
        return ArchSearch(lambda cfg: {"score": 0}, {"x": (1.0, 10.0)},
                          backend="none", log_path=tmp_log, **kw)

    def test_curve_labels_include_run_id(self, tmp_log):
        history = [{"id": 7, "status": "ok", "score": 1.0, "config": {"x": 2.0},
                    "curves": {"train_losses": [1.0, 0.5], "val_losses": [1.1, 0.8]}}]
        assert "#7 x=2.000:" in self._search(tmp_log)._build_prompt(history)

    def test_missing_score_rendered_as_dash(self, tmp_log):
        history = [{"id": 1, "status": "error", "config": {"x": 2.0}}]
        prompt = self._search(tmp_log)._build_prompt(history)
        assert "| 1 | 2 | - |" in prompt

    def test_resume_banner_says_score(self, tmp_log, capsys):
        with open(tmp_log, "w") as f:
            f.write(json.dumps({"id": 1, "config": {"x": 5.0}, "score": 0.9, "status": "ok"}) + "\n")
        self._search(tmp_log, minimize=False).run(max_evals=1)
        out = capsys.readouterr().out
        assert "Best so far: score=" in out
        assert "loss=" not in out


class TestFitSignal:
    def test_overfitting_on_mse_scale(self):
        from neuropt.arch_search import _fit_signal
        assert _fit_signal([100.0, 60.0, 30.0], [110.0, 80.0, 95.0]) == "overfitting"

    def test_overfitting_on_small_scale(self):
        from neuropt.arch_search import _fit_signal
        assert _fit_signal([0.10, 0.05, 0.02], [0.11, 0.06, 0.08]) == "overfitting"

    def test_underfitting_is_relative(self):
        from neuropt.arch_search import _fit_signal
        assert _fit_signal([0.30, 0.29, 0.285], [0.31, 0.30, 0.29]) == "underfitting"
        assert _fit_signal([3.0, 1.0, 0.6], [3.1, 1.1, 0.7]) is None


class TestAutoQwenNotice:
    def test_notice_when_falling_back_to_qwen(self, monkeypatch, capsys):
        import neuropt.backends as backends

        class Avail(BaseLLMBackend):
            def __init__(self, ok):
                super().__init__()
                self.ok = ok

            def generate(self, prompt, max_tokens=1024):
                return ""

            def is_available(self):
                return self.ok

        monkeypatch.setattr(backends, "_REGISTRY", {
            "claude": lambda: Avail(False),
            "openai": lambda: Avail(False),
            "qwen": lambda: Avail(True),
        })
        assert backends.get_default_backend() is not None
        assert "Qwen2.5-1.5B" in capsys.readouterr().out
