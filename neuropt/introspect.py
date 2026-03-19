"""Introspect a model to auto-generate a search space.

Supports PyTorch nn.Module and sklearn-compatible models (XGBoost, LightGBM, etc).
Detects pretrained weights and generates fine-tuning search spaces.
"""

import copy
import json
import math
import re

from neuropt.search_space import Categorical, IntUniform, LogUniform, Uniform


# Activation types we know how to swap
ACTIVATION_TYPES = None  # lazy import to avoid hard torch dependency


def _get_act_types():
    global ACTIVATION_TYPES
    if ACTIVATION_TYPES is None:
        import torch.nn as nn
        ACTIVATION_TYPES = (
            nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.ELU, nn.Tanh,
            nn.Mish, nn.Hardswish, nn.PReLU,
        )
    return ACTIVATION_TYPES


def _get_act_cls(name):
    import torch.nn as nn
    return {
        "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
        "leaky_relu": nn.LeakyReLU, "mish": nn.Mish,
        "hardswish": nn.Hardswish, "prelu": nn.PReLU,
    }.get(name, nn.ReLU)


# ── Attention pooling module ─────────────────────────────────────────────

class AttentionPool2d:
    """Lazy-loaded to avoid import torch at module level."""
    _cls = None

    @classmethod
    def get_cls(cls):
        if cls._cls is None:
            import torch
            import torch.nn as nn

            class AttentionPool2dModule(nn.Module):
                """Learned attention pooling over spatial dimensions.

                Replaces AdaptiveAvgPool/MaxPool with a weighted average where
                the weights are learned via a linear projection + softmax.
                Output shape is identical: (B, C, 1, 1).
                """
                def __init__(self, channels):
                    super().__init__()
                    self.query = nn.Linear(channels, 1, bias=False)

                def forward(self, x):
                    # x: (B, C, H, W) → flatten spatial → attend → pool
                    B, C, H, W = x.shape
                    flat = x.view(B, C, H * W).permute(0, 2, 1)  # (B, HW, C)
                    scores = self.query(flat).squeeze(-1)          # (B, HW)
                    weights = torch.softmax(scores, dim=-1)        # (B, HW)
                    pooled = (flat * weights.unsqueeze(-1)).sum(dim=1)  # (B, C)
                    return pooled.view(B, C, 1, 1)

            cls._cls = AttentionPool2dModule
        return cls._cls


def _classify_dropout_path(path: str) -> str:
    """Classify a dropout layer by its path name for per-path grouping."""
    lower = path.lower()
    if "attn" in lower or "attention" in lower:
        return "attn"
    if "ff" in lower or "ffn" in lower or "mlp" in lower:
        return "ff"
    if "embed" in lower:
        return "embed"
    return "default"


def _detect_pretrained(model) -> bool:
    """Detect if weights are pretrained by checking parameter statistics.

    Trained weights have smaller variance than random init. PyTorch's default
    Linear init uses kaiming_uniform_(a=sqrt(5)) which gives variance =
    1/(3*fan_in). Pretrained models after training with weight decay typically
    have much less variance than this.
    """
    param_ratios = []
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # skip biases and 1-d params
        fan_in = param.shape[1]
        for d in param.shape[2:]:
            fan_in *= d
        # PyTorch default init variance: kaiming_uniform_(a=sqrt(5)) → 1/(3*fan_in)
        expected_var = 1.0 / (3.0 * fan_in)
        actual_var = param.data.var().item()
        if expected_var > 0 and actual_var > 0:
            param_ratios.append(actual_var / expected_var)

    if not param_ratios:
        return False

    # Random init gives ratio ~1.0; pretrained models typically 0.01-0.3
    median_ratio = sorted(param_ratios)[len(param_ratios) // 2]
    return median_ratio < 0.5


def _find_layer_groups(model):
    """Find repeating layer groups (e.g., transformer blocks, ResNet layers).

    Returns a list of (group_name, [module_paths]) tuples.
    """
    import torch.nn as nn

    groups = []
    for name, mod in model.named_modules():
        # Skip the root module
        if name == "":
            continue
        # Look for numbered children (layer.0, layer.1, etc.)
        children = list(mod.named_children())
        if len(children) >= 2:
            numbered = [c for c in children if c[0].isdigit()]
            if len(numbered) >= 2:
                paths = [f"{name}.{c[0]}" if name else c[0] for c in numbered]
                groups.append((name, paths))

    return groups


def _find_last_linear(model):
    """Find the last nn.Linear module path (typically the classification head)."""
    import torch.nn as nn

    last_linear_path = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            last_linear_path = name
    return last_linear_path


def introspect(model, pretrained=None):
    """Walk a model's module tree and find tunable components.

    Returns a dict describing what was found: activation types, dropout
    layers and rates, batch norm presence, layer norm, pretrained status,
    layer groups, and module paths for each.

    Args:
        model: PyTorch nn.Module to introspect.
        pretrained: Override pretrained detection. If None, auto-detect.
    """
    import torch.nn as nn
    act_types = _get_act_types()

    # All dropout types we detect
    _dropout_types = (
        nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
        nn.AlphaDropout, nn.FeatureAlphaDropout,
    )

    # Adaptive pooling types we can swap between
    _adaptive_pool_types = (
        nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
        nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    )

    info = {
        "activations_found": set(),
        "activation_paths": [],
        "has_dropout": False,
        "dropout_rate": 0.0,
        "dropout_paths": [],
        "dropout_groups": {},  # group_name -> [paths]
        "mha_dropout_paths": [],  # MultiheadAttention internal dropout
        "has_batchnorm": False,
        "batchnorm_paths": [],
        "has_layernorm": False,
        "layernorm_paths": [],
        "has_pool": False,
        "pool_paths": [],  # adaptive pool layers
        "pool_type": None,  # "avg" or "max"
        "has_conv": False,
        "has_linear": False,
        "n_params": sum(p.numel() for p in model.parameters()),
        "is_pretrained": False,
        "layer_groups": [],
        "last_linear_path": None,
    }

    for name, mod in model.named_modules():
        if isinstance(mod, act_types):
            info["activations_found"].add(type(mod).__name__)
            info["activation_paths"].append(name)
        elif isinstance(mod, _dropout_types):
            info["has_dropout"] = True
            if not info["dropout_paths"]:  # keep the first rate as representative
                info["dropout_rate"] = mod.p
            info["dropout_paths"].append(name)
            # Classify dropout by path
            group = _classify_dropout_path(name)
            info["dropout_groups"].setdefault(group, []).append(name)
        elif isinstance(mod, nn.MultiheadAttention):
            # MHA has an internal dropout float, not a module — track it
            if mod.dropout > 0:
                info["mha_dropout_paths"].append(name)
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            info["has_batchnorm"] = True
            info["batchnorm_paths"].append(name)
        elif isinstance(mod, nn.LayerNorm):
            info["has_layernorm"] = True
            info["layernorm_paths"].append(name)
        elif isinstance(mod, _adaptive_pool_types):
            info["has_pool"] = True
            info["pool_paths"].append(name)
            if "Max" in type(mod).__name__:
                info["pool_type"] = "max"
            else:
                info["pool_type"] = "avg"
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            info["has_conv"] = True
        elif isinstance(mod, nn.Linear):
            info["has_linear"] = True

    # Pretrained detection
    if pretrained is not None:
        info["is_pretrained"] = pretrained
    else:
        info["is_pretrained"] = _detect_pretrained(model)

    # Layer groups and last linear (for freeze strategies)
    info["layer_groups"] = _find_layer_groups(model)
    info["last_linear_path"] = _find_last_linear(model)

    return info


def build_search_space(info):
    """Generate a search space from introspection results."""
    space = {}

    if info["activation_paths"]:
        space["activation"] = Categorical([
            "relu", "gelu", "silu", "leaky_relu", "mish", "hardswish", "prelu",
        ])

    # Per-path dropout: emit separate params if multiple groups found
    if info["has_dropout"]:
        groups = info.get("dropout_groups", {})
        non_default_groups = {k: v for k, v in groups.items() if k != "default"}
        if len(non_default_groups) >= 2:
            # Multiple distinct groups — emit per-path dropout params
            for group_name in sorted(groups.keys()):
                if group_name == "default":
                    space["dropout"] = Uniform(0.0, 0.5)
                else:
                    space[f"dropout_{group_name}"] = Uniform(0.0, 0.5)
        else:
            space["dropout"] = Uniform(0.0, 0.5)

    if info["has_batchnorm"]:
        space["use_batchnorm"] = Categorical([True, False])

    if info.get("has_layernorm"):
        space["use_layernorm"] = Categorical([True, False])

    if info.get("has_pool"):
        space["pool_type"] = Categorical(["avg", "max", "attention"])

    if info.get("mha_dropout_paths"):
        space["mha_dropout"] = Uniform(0.0, 0.3)

    # Always include training hyperparams
    space["lr"] = LogUniform(1e-4, 1e-1)
    space["wd"] = LogUniform(1e-6, 1e-2)
    space["optimizer"] = Categorical(["sgd", "adam", "adamw"])

    # Fine-tuning params for pretrained models
    if info.get("is_pretrained"):
        space["freeze_strategy"] = Categorical([
            "full", "head_only", "gradual_unfreeze", "all_but_embeddings",
        ])
        space["lr_layer_decay"] = Uniform(0.5, 1.0)
        space["l2sp_regularization"] = Categorical([True, False])

    return space


def build_ml_context(info):
    """Generate LLM context describing what was found in the model."""
    parts = ["You are optimizing an existing PyTorch model.\n"]
    parts.append("## What was detected in the model\n")

    if info["activations_found"]:
        parts.append(f"- Activation functions: {', '.join(sorted(info['activations_found']))}")
        parts.append(f"  ({len(info['activation_paths'])} swappable activation layers)")
    if info["has_dropout"]:
        groups = info.get("dropout_groups", {})
        non_default = {k: v for k, v in groups.items() if k != "default"}
        if len(non_default) >= 2:
            group_desc = ", ".join(f"{k}({len(v)})" for k, v in sorted(groups.items()))
            parts.append(f"- Dropout layers by path: {group_desc}")
        else:
            parts.append(f"- Dropout layers found (original rate: {info['dropout_rate']:.2f})")
    if info["has_batchnorm"]:
        parts.append(f"- Batch normalization: {len(info['batchnorm_paths'])} layers (can be toggled off)")
    if info.get("has_layernorm"):
        parts.append(f"- Layer normalization: {len(info['layernorm_paths'])} layers (can be toggled off)")
    if info.get("has_pool"):
        pool_desc = info.get("pool_type", "unknown")
        parts.append(f"- Adaptive pooling: {len(info['pool_paths'])} layers (current: {pool_desc})")
        parts.append("  Swappable: avg, max, or learned attention pooling")
    if info.get("mha_dropout_paths"):
        parts.append(f"- MultiheadAttention internal dropout: {len(info['mha_dropout_paths'])} layers")
    if info["has_conv"]:
        parts.append("- Convolutional layers present")
    if info["has_linear"]:
        parts.append("- Fully connected layers present")
    if info.get("is_pretrained"):
        parts.append("- **Pretrained model detected** — fine-tuning strategies available")
        if info.get("layer_groups"):
            n_groups = sum(len(paths) for _, paths in info["layer_groups"])
            parts.append(f"  Layer groups: {n_groups} blocks across {len(info['layer_groups'])} groups")
        if info.get("last_linear_path"):
            parts.append(f"  Classification head: {info['last_linear_path']}")
    parts.append(f"- Total parameters: {info['n_params']:,}")
    parts.append("")

    parts.append("## Guidance\n")
    parts.append(
        "- GELU and SiLU often outperform ReLU; Mish is competitive with SiLU\n"
        "- Hardswish is efficient on mobile/edge — similar to SiLU but cheaper\n"
        "- PReLU has a learnable slope — can help in face recognition and similar tasks\n"
        "- If overfitting: increase dropout, increase weight decay\n"
        "- If underfitting: reduce dropout (even to 0), reduce weight decay\n"
        "- BatchNorm usually helps but can sometimes hurt very small models\n"
        "- LayerNorm is standard in transformers — toggling off is rarely helpful but worth testing\n"
        "- Attention pooling learns which spatial positions matter — try it if avg/max pool plateau\n"
        "- Max pooling keeps strongest activations; avg pooling considers all features\n"
        "- AdamW with lr ~1e-3 is a safe starting point\n"
        "- SGD needs higher lr (0.01-0.1) and is more sensitive to tuning\n"
        "- Read the training curves: train-val gap = overfitting, both stuck = underfitting\n"
        "- Balance exploration with exploitation — try new things but also refine what works"
    )

    if info.get("is_pretrained"):
        parts.append("")
        parts.append("## Fine-tuning guidance\n")
        parts.append(
            "- head_only: safest for small datasets, only trains the classification head\n"
            "- gradual_unfreeze: good default for medium datasets, unfreezes last ~1/3 of layers\n"
            "- full: trains everything — best with enough data, risk of forgetting\n"
            "- all_but_embeddings: freezes only embedding layers, good for NLP models\n"
            "- lr_layer_decay near 0.5 = aggressive (lower layers learn much slower)\n"
            "- lr_layer_decay near 1.0 = uniform learning rate across all layers\n"
            "- l2sp_regularization: regularizes toward pretrained weights instead of zero,\n"
            "  prevents catastrophic forgetting — try if full fine-tuning overfits"
        )

    return "\n".join(parts)


def apply_config(model, config, info):
    """Modify a deep-copied model in-place based on a config dict."""
    import torch.nn as nn

    # Swap activations
    if "activation" in config and info["activation_paths"]:
        act_cls = _get_act_cls(config["activation"])
        for path in info["activation_paths"]:
            _set_module(model, path, act_cls())

    # Set dropout rates — per-path or global
    groups = info.get("dropout_groups", {})
    non_default = {k: v for k, v in groups.items() if k != "default"}
    if len(non_default) >= 2:
        # Per-path dropout mode
        for group_name, paths in groups.items():
            key = "dropout" if group_name == "default" else f"dropout_{group_name}"
            if key in config:
                for path in paths:
                    mod = _get_module(model, path)
                    mod.p = config[key]
    elif "dropout" in config:
        for path in info["dropout_paths"]:
            mod = _get_module(model, path)
            mod.p = config["dropout"]

    # Toggle batch norm off
    if "use_batchnorm" in config and not config["use_batchnorm"]:
        for path in info["batchnorm_paths"]:
            _set_module(model, path, nn.Identity())

    # Toggle layer norm off
    if "use_layernorm" in config and not config["use_layernorm"]:
        for path in info.get("layernorm_paths", []):
            _set_module(model, path, nn.Identity())

    # Swap pooling type
    if "pool_type" in config and info.get("pool_paths"):
        _swap_pool(model, config["pool_type"], info)

    # Set MHA internal dropout
    if "mha_dropout" in config:
        for path in info.get("mha_dropout_paths", []):
            mod = _get_module(model, path)
            mod.dropout = config["mha_dropout"]

    # Apply freeze strategies for pretrained models
    if info.get("is_pretrained") and "freeze_strategy" in config:
        _apply_freeze_strategy(model, config["freeze_strategy"], info)


def _swap_pool(model, pool_type, info):
    """Swap adaptive pooling layers to avg, max, or attention."""
    import torch.nn as nn

    for path in info["pool_paths"]:
        old = _get_module(model, path)
        # Get the output_size from the existing pool
        output_size = old.output_size

        if pool_type == "avg":
            # Determine dimensionality from old module type
            name = type(old).__name__
            if "1d" in name:
                new = nn.AdaptiveAvgPool1d(output_size)
            elif "3d" in name:
                new = nn.AdaptiveAvgPool3d(output_size)
            else:
                new = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == "max":
            name = type(old).__name__
            if "1d" in name:
                new = nn.AdaptiveMaxPool1d(output_size)
            elif "3d" in name:
                new = nn.AdaptiveMaxPool3d(output_size)
            else:
                new = nn.AdaptiveMaxPool2d(output_size)
        elif pool_type == "attention":
            # Need to figure out channel count from surrounding context
            channels = _infer_channels_before_pool(model, path)
            if channels is not None:
                new = AttentionPool2d.get_cls()(channels)
            else:
                import warnings
                warnings.warn(
                    f"Cannot infer channel count for attention pooling at '{path}', "
                    f"keeping original pooling layer"
                )
                continue
        else:
            continue
        _set_module(model, path, new)


def _infer_channels_before_pool(model, pool_path):
    """Infer the number of channels feeding into a pooling layer."""
    import torch.nn as nn

    # Walk all modules, track the last conv/bn channel count before the pool
    last_channels = None
    for name, mod in model.named_modules():
        if name == pool_path:
            break
        if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            last_channels = mod.out_channels
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            last_channels = mod.num_features
    return last_channels


def _apply_freeze_strategy(model, strategy, info):
    """Apply a freeze strategy to a pretrained model."""
    import torch.nn as nn

    if strategy == "full":
        # Train everything — no-op
        return

    if strategy == "head_only":
        # Freeze all, then unfreeze last Linear
        for param in model.parameters():
            param.requires_grad = False
        last_path = info.get("last_linear_path")
        if last_path:
            last_mod = _get_module(model, last_path)
            for param in last_mod.parameters():
                param.requires_grad = True

    elif strategy == "gradual_unfreeze":
        # Freeze all, then unfreeze last ~1/3 of layer groups
        for param in model.parameters():
            param.requires_grad = False
        layer_groups = info.get("layer_groups", [])
        if layer_groups:
            # Use the largest group (most likely the main backbone)
            _, paths = max(layer_groups, key=lambda g: len(g[1]))
            n_unfreeze = max(1, len(paths) // 3)
            for path in paths[-n_unfreeze:]:
                mod = _get_module(model, path)
                for param in mod.parameters():
                    param.requires_grad = True
        # Always unfreeze the head
        last_path = info.get("last_linear_path")
        if last_path:
            last_mod = _get_module(model, last_path)
            for param in last_mod.parameters():
                param.requires_grad = True

    elif strategy == "all_but_embeddings":
        # Freeze only embedding layers
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Embedding):
                for param in mod.parameters():
                    param.requires_grad = False


def make_wrapped_train_fn(model, train_fn, info):
    """Create a train_fn that deep-copies the model and applies config modifications.

    The user's train_fn receives ``config["model"]`` (the modified model)
    plus any search space keys (activation, dropout, lr, wd, optimizer, etc).

    For pretrained models with L2-SP regularization enabled, snapshots the
    pretrained weights once and injects ``config["pretrained_weights"]``.
    """
    # Snapshot pretrained weights once for L2-SP
    pretrained_weights = None
    if info.get("is_pretrained"):
        pretrained_weights = {
            name: param.data.clone()
            for name, param in model.named_parameters()
        }

    def wrapped(config):
        modified = copy.deepcopy(model)
        apply_config(modified, config, info)
        config_with_model = dict(config)
        config_with_model["model"] = modified
        # Inject pretrained weights when L2-SP is requested
        if config.get("l2sp_regularization") and pretrained_weights is not None:
            config_with_model["pretrained_weights"] = pretrained_weights
        return train_fn(config_with_model)
    return wrapped


# ── Module path helpers ──────────────────────────────────────────────────

def _get_module(model, path):
    """Get a module by dot-separated path (handles Sequential integer indices)."""
    parts = path.split(".")
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def _set_module(model, path, new_module):
    """Replace a module at the given dot-separated path."""
    parts = path.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)

    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


# ── Sklearn-compatible model introspection ───────────────────────────────

SKIP_PARAMS = {
    "random_state", "seed", "n_jobs", "verbose", "verbosity", "silent",
    "objective", "eval_metric", "use_label_encoder", "device", "gpu_id",
    "tree_method", "predictor", "booster", "importance_type", "callbacks",
    "enable_categorical", "feature_types", "max_cat_to_onehot",
    "max_cat_threshold", "interaction_constraints", "monotone_constraints",
    "base_score", "validate_parameters", "nthread",
}


def is_sklearn_compatible(model):
    """Check if a model has sklearn-style get_params/set_params."""
    return hasattr(model, "get_params") and hasattr(model, "set_params")


def introspect_sklearn(model):
    """Introspect an sklearn-compatible model (XGBoost, LightGBM, sklearn, etc).

    Returns a dict with model class name and tunable parameters + current values.
    Includes None-valued params since many libraries use None as "use internal default."
    """
    params = model.get_params()
    tunable = {}
    for name, value in params.items():
        if name in SKIP_PARAMS:
            continue
        if isinstance(value, (int, float, bool, str)):
            tunable[name] = value
        elif value is None:
            tunable[name] = None  # include — LLM knows the real defaults

    return {
        "model_type": type(model).__name__,
        "model_module": type(model).__module__,
        "all_params": params,
        "tunable_params": tunable,
    }


def build_sklearn_search_space_with_llm(info, backend):
    """Ask the LLM for reasonable search ranges given the model type and params."""
    model_type = info["model_type"]
    tunable = info["tunable_params"]

    prompt = (
        f"You are setting up a hyperparameter search for a {model_type} model.\n\n"
        f"Here are its tunable parameters and current values:\n"
    )
    for name, value in tunable.items():
        prompt += f"  {name} = {value!r} ({type(value).__name__})\n"

    prompt += (
        f"\nFor each parameter, provide a search range as JSON. Use this format:\n"
        f'{{"param_name": {{"type": "int"|"float"|"log_float"|"bool"|"choice", '
        f'"min": ..., "max": ..., "choices": [...]}}}}\n\n'
        f"Rules:\n"
        f"- Only include parameters worth tuning (skip ones that rarely matter)\n"
        f"- Use \"log_float\" for parameters that span orders of magnitude (like learning_rate, reg_alpha, reg_lambda)\n"
        f"- Use \"int\" for integer parameters with a range\n"
        f"- Use \"float\" for bounded float parameters\n"
        f"- Use \"bool\" for boolean toggles\n"
        f"- Use \"choice\" for categorical options\n"
        f"- Choose ranges that a practitioner would actually search over\n\n"
        f"Respond with ONLY the JSON object. No explanation."
    )

    response = backend.generate(prompt, max_tokens=1024)

    # Parse response
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if not match:
        return _fallback_sklearn_search_space(info)

    try:
        ranges = json.loads(match.group())
    except json.JSONDecodeError:
        return _fallback_sklearn_search_space(info)

    space = {}
    for name, spec in ranges.items():
        if name not in tunable:
            continue
        try:
            t = spec.get("type", "float")
            if t == "log_float":
                lo, hi = float(spec["min"]), float(spec["max"])
                if lo <= 0:
                    lo = 1e-8
                if hi <= lo:
                    hi = lo * 100
                space[name] = LogUniform(lo, hi)
            elif t == "float":
                space[name] = Uniform(float(spec["min"]), float(spec["max"]))
            elif t == "int":
                space[name] = IntUniform(int(spec["min"]), int(spec["max"]))
            elif t == "bool":
                space[name] = Categorical([True, False])
            elif t == "choice":
                space[name] = Categorical(spec["choices"])
        except (KeyError, TypeError, ValueError):
            continue

    if not space:
        return _fallback_sklearn_search_space(info)

    return space


_KNOWN_RANGES = {
    "max_depth":        IntUniform(3, 12),
    "n_estimators":     IntUniform(50, 500),
    "learning_rate":    LogUniform(1e-3, 0.3),
    "eta":              LogUniform(1e-3, 0.3),
    "min_child_weight": IntUniform(1, 10),
    "subsample":        Uniform(0.5, 1.0),
    "colsample_bytree": Uniform(0.5, 1.0),
    "colsample_bylevel": Uniform(0.5, 1.0),
    "colsample_bynode": Uniform(0.5, 1.0),
    "gamma":            LogUniform(1e-5, 10.0),
    "reg_alpha":        LogUniform(1e-5, 10.0),
    "reg_lambda":       LogUniform(1e-5, 10.0),
    "max_delta_step":   IntUniform(0, 10),
    "scale_pos_weight": Uniform(0.5, 5.0),
    "max_leaves":       IntUniform(0, 128),
    "num_leaves":       IntUniform(15, 127),
    "min_data_in_leaf": IntUniform(5, 100),
    "bagging_fraction": Uniform(0.5, 1.0),
    "feature_fraction": Uniform(0.5, 1.0),
    "max_features":     Uniform(0.5, 1.0),
    "min_samples_split": IntUniform(2, 20),
    "min_samples_leaf": IntUniform(1, 20),
    "max_samples":      Uniform(0.5, 1.0),
}


def _fallback_sklearn_search_space(info):
    """Heuristic-based search space when LLM isn't available or fails."""
    space = {}
    for name, value in info["tunable_params"].items():
        if name in _KNOWN_RANGES:
            space[name] = _KNOWN_RANGES[name]
        elif isinstance(value, bool):
            space[name] = Categorical([True, False])
        elif isinstance(value, int) and value > 0:
            space[name] = IntUniform(max(1, value // 3), value * 3)
        elif isinstance(value, float) and value > 0:
            if value < 0.01 or "reg" in name or "alpha" in name or "lambda" in name:
                space[name] = LogUniform(value / 10, min(value * 10, 100))
            elif value <= 1.0:
                space[name] = Uniform(max(0.0, value - 0.3), min(1.0, value + 0.3))
            else:
                space[name] = Uniform(value / 3, value * 3)
    return space


def build_sklearn_ml_context(info, space):
    """Generate LLM context for an sklearn-compatible model."""
    parts = [f"You are optimizing a {info['model_type']} model.\n"]
    parts.append("## Parameters being searched\n")
    for name, dim in space.items():
        current = info["tunable_params"].get(name, "?")
        parts.append(f"- {name} (current: {current!r}): {dim}")
    parts.append("")
    parts.append(
        "## Guidance\n"
        "- Read the training curves to spot overfitting vs underfitting\n"
        "- If overfitting: increase regularization, reduce model complexity\n"
        "- If underfitting: reduce regularization, increase complexity\n"
        "- Balance exploration with exploitation\n"
        "- Don't repeat configs that have already been tried"
    )
    return "\n".join(parts)


def make_sklearn_wrapped_train_fn(model, train_fn):
    """Wrap train_fn to clone the model and set params from config each call."""
    def wrapped(config):
        # Separate search params from non-model keys
        model_params = {}
        extra = {}
        param_names = set(model.get_params().keys())
        for k, v in config.items():
            if k in param_names:
                model_params[k] = v
            else:
                extra[k] = v

        from sklearn.base import clone
        cloned = clone(model)
        cloned.set_params(**model_params)

        config_with_model = dict(config)
        config_with_model["model"] = cloned
        return train_fn(config_with_model)

    return wrapped
