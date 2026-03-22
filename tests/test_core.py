"""Core unit tests for neuropt — no LLM needed, all run with backend='none'."""

import copy
import json
import math
import os
import tempfile

import pytest
import torch
import torch.nn as nn

from neuropt import ArchSearch, Categorical, IntUniform, LogUniform, Uniform
from neuropt.backends.base import BaseLLMBackend
from neuropt.introspect import (
    apply_config,
    build_ml_context,
    build_search_space,
    introspect,
    make_wrapped_train_fn,
    _detect_pretrained,
    _classify_dropout_path,
    _find_layer_groups,
    _find_last_linear,
    AttentionPool2d,
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
