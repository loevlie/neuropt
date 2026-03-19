"""Introspect a PyTorch model to auto-generate a search space."""

import copy

from neuropt.search_space import Categorical, LogUniform, Uniform


# Activation types we know how to swap
ACTIVATION_TYPES = None  # lazy import to avoid hard torch dependency
ACTIVATION_MAP = {
    "relu": "ReLU",
    "gelu": "GELU",
    "silu": "SiLU",
    "leaky_relu": "LeakyReLU",
}


def _get_act_types():
    global ACTIVATION_TYPES
    if ACTIVATION_TYPES is None:
        import torch.nn as nn
        ACTIVATION_TYPES = (nn.ReLU, nn.GELU, nn.SiLU, nn.LeakyReLU, nn.ELU, nn.Tanh)
    return ACTIVATION_TYPES


def _get_act_cls(name):
    import torch.nn as nn
    return {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
            "leaky_relu": nn.LeakyReLU}.get(name, nn.ReLU)


def introspect(model):
    """Walk a model's module tree and find tunable components.

    Returns a dict describing what was found: activation types, dropout
    layers and rates, batch norm presence, and module paths for each.
    """
    import torch.nn as nn
    act_types = _get_act_types()

    info = {
        "activations_found": set(),
        "activation_paths": [],
        "has_dropout": False,
        "dropout_rate": 0.0,
        "dropout_paths": [],
        "has_batchnorm": False,
        "batchnorm_paths": [],
        "has_conv": False,
        "has_linear": False,
        "n_params": sum(p.numel() for p in model.parameters()),
    }

    for name, mod in model.named_modules():
        if isinstance(mod, act_types):
            info["activations_found"].add(type(mod).__name__)
            info["activation_paths"].append(name)
        elif isinstance(mod, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            info["has_dropout"] = True
            info["dropout_rate"] = mod.p
            info["dropout_paths"].append(name)
        elif isinstance(mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            info["has_batchnorm"] = True
            info["batchnorm_paths"].append(name)
        elif isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            info["has_conv"] = True
        elif isinstance(mod, nn.Linear):
            info["has_linear"] = True

    return info


def build_search_space(info):
    """Generate a search space from introspection results."""
    space = {}

    if info["activation_paths"]:
        space["activation"] = Categorical(["relu", "gelu", "silu", "leaky_relu"])

    if info["has_dropout"]:
        space["dropout"] = Uniform(0.0, 0.5)

    if info["has_batchnorm"]:
        space["use_batchnorm"] = Categorical([True, False])

    # Always include training hyperparams
    space["lr"] = LogUniform(1e-4, 1e-1)
    space["wd"] = LogUniform(1e-6, 1e-2)
    space["optimizer"] = Categorical(["sgd", "adam", "adamw"])

    return space


def build_ml_context(info):
    """Generate LLM context describing what was found in the model."""
    parts = ["You are optimizing an existing PyTorch model.\n"]
    parts.append("## What was detected in the model\n")

    if info["activations_found"]:
        parts.append(f"- Activation functions: {', '.join(sorted(info['activations_found']))}")
        parts.append(f"  ({len(info['activation_paths'])} swappable activation layers)")
    if info["has_dropout"]:
        parts.append(f"- Dropout layers found (original rate: {info['dropout_rate']:.2f})")
    if info["has_batchnorm"]:
        parts.append(f"- Batch normalization: {len(info['batchnorm_paths'])} layers (can be toggled off)")
    if info["has_conv"]:
        parts.append("- Convolutional layers present")
    if info["has_linear"]:
        parts.append("- Fully connected layers present")
    parts.append(f"- Total parameters: {info['n_params']:,}")
    parts.append("")

    parts.append("## Guidance\n")
    parts.append(
        "- GELU and SiLU often outperform ReLU — try them\n"
        "- If overfitting: increase dropout, increase weight decay\n"
        "- If underfitting: reduce dropout (even to 0), reduce weight decay\n"
        "- BatchNorm usually helps but can sometimes hurt very small models\n"
        "- AdamW with lr ~1e-3 is a safe starting point\n"
        "- SGD needs higher lr (0.01-0.1) and is more sensitive to tuning\n"
        "- Read the training curves: train-val gap = overfitting, both stuck = underfitting\n"
        "- Balance exploration with exploitation — try new things but also refine what works"
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

    # Set dropout rates
    if "dropout" in config:
        for path in info["dropout_paths"]:
            mod = _get_module(model, path)
            mod.p = config["dropout"]

    # Toggle batch norm off
    if "use_batchnorm" in config and not config["use_batchnorm"]:
        for path in info["batchnorm_paths"]:
            _set_module(model, path, nn.Identity())


def make_wrapped_train_fn(model, train_fn, info):
    """Create a train_fn that deep-copies the model and applies config modifications.

    The user's train_fn receives ``config["model"]`` (the modified model)
    plus any search space keys (activation, dropout, lr, wd, optimizer, etc).
    """
    def wrapped(config):
        modified = copy.deepcopy(model)
        apply_config(modified, config, info)
        config_with_model = dict(config)
        config_with_model["model"] = modified
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
